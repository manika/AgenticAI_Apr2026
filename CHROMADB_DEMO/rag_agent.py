from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from chroma_client import DEFAULT_COLLECTION, DEFAULT_TOP_K, get_llm, get_vectorstore

MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Use the provided context to answer the question. "
    "If the context does not contain the answer, say you don't know and suggest what to "
    "ingest to answer better."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str               # original user question (never mutated)
    rewritten_question: str     # current retrieval query (updated on rewrites)
    docs: List[Document]        # raw retrieved docs
    filtered_docs: List[Document]  # relevance-graded subset
    answer: str                 # generated answer
    retry_count: int            # shared retry counter
    grounded: bool              # True when grade_generation passes


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------

class GradeDoc(BaseModel):
    score: Literal["yes", "no"]


class GradeGeneration(BaseModel):
    grounded: Literal["yes", "no"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(docs: List[Document]) -> str:
    if not docs:
        return ""
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        source = (doc.metadata or {}).get("source", "unknown")
        chunks.append(f"[{idx}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def get_graph(collection_name: str = DEFAULT_COLLECTION):
    retriever = get_vectorstore(collection_name).as_retriever(
        search_kwargs={"k": DEFAULT_TOP_K}
    )
    llm = get_llm()
    grader_llm = llm.with_structured_output(GradeDoc)
    generation_grader_llm = llm.with_structured_output(GradeGeneration)

    # ------------------------------------------------------------------
    # Node 1: rewrite_query
    # Rewrites the question for better vector search retrieval.
    # On first call initialises retry_count; on retries increments it.
    # ------------------------------------------------------------------
    def rewrite_query(state: RAGState) -> dict:
        retry_count = state.get("retry_count", -1) + 1
        question = state["question"]
        response = llm.invoke(
            [
                SystemMessage(
                    "You are a query optimisation assistant. Rewrite the user's question "
                    "to be more specific and concrete so that a vector similarity search "
                    "returns more relevant results. Output only the rewritten question, "
                    "no explanation."
                ),
                HumanMessage(question),
            ]
        )
        rewritten = response.content.strip()
        print(f"[rewrite_query] retry={retry_count} → '{rewritten}'")
        return {
            "rewritten_question": rewritten,
            "retry_count": retry_count,
        }

    # ------------------------------------------------------------------
    # Node 2: retrieve
    # Retrieves top-k docs using the (possibly rewritten) question.
    # ------------------------------------------------------------------
    def retrieve(state: RAGState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        docs = retriever.invoke(query)
        print(f"[retrieve] fetched {len(docs)} docs")
        return {"docs": docs}

    # ------------------------------------------------------------------
    # Node 3: grade_documents
    # Grades each retrieved doc for relevance; filters irrelevant ones.
    # ------------------------------------------------------------------
    def grade_documents(state: RAGState) -> dict:
        question = state.get("rewritten_question") or state["question"]
        docs = state.get("docs", [])
        filtered: List[Document] = []
        for doc in docs:
            result: GradeDoc = grader_llm.invoke(
                [
                    SystemMessage(
                        "You are a relevance grader. Assess whether the document is "
                        "relevant to the question. Reply with 'yes' or 'no' only."
                    ),
                    HumanMessage(
                        f"Question: {question}\n\nDocument:\n{doc.page_content[:1000]}"
                    ),
                ]
            )
            if result.score == "yes":
                filtered.append(doc)
        print(
            f"[grade_documents] {len(filtered)}/{len(docs)} docs are relevant"
        )
        return {"filtered_docs": filtered}

    # ------------------------------------------------------------------
    # Node 4: generate
    # Generates a streamed answer from filtered_docs (or all docs).
    # ------------------------------------------------------------------
    async def generate(state: RAGState) -> dict:
        writer = get_stream_writer()
        context_docs = state.get("filtered_docs") or state.get("docs", [])
        context = _format_context(context_docs)
        question = state["question"]
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context if context else '[no context retrieved]'}"
        )
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_prompt)]

        parts: List[str] = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                parts.append(chunk.content)
                writer({"type": "token", "content": chunk.content})

        return {"answer": "".join(parts)}

    # ------------------------------------------------------------------
    # Node 5: grade_generation
    # Checks whether the answer is grounded in the retrieved context.
    # ------------------------------------------------------------------
    def grade_generation(state: RAGState) -> dict:
        context_docs = state.get("filtered_docs") or state.get("docs", [])
        context = _format_context(context_docs)
        answer = state.get("answer", "")
        result: GradeGeneration = generation_grader_llm.invoke(
            [
                SystemMessage(
                    "You are a hallucination grader. Assess whether the answer is "
                    "fully grounded in and supported by the provided context. "
                    "Reply with 'yes' if grounded, 'no' if it contains unsupported claims."
                ),
                HumanMessage(
                    f"Context:\n{context[:3000]}\n\nAnswer:\n{answer}"
                ),
            ]
        )
        grounded = result.grounded == "yes"
        print(f"[grade_generation] grounded={grounded}")
        return {"grounded": grounded}

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------
    def route_after_grade_docs(state: RAGState) -> str:
        if state.get("filtered_docs") or state.get("retry_count", 0) >= MAX_RETRIES:
            return "generate"
        return "rewrite_query"

    def route_after_grade_generation(state: RAGState) -> str:
        if state.get("grounded") or state.get("retry_count", 0) >= MAX_RETRIES:
            return END
        return "rewrite_query"

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("grade_generation", grade_generation)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade_docs,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )

    graph.add_edge("generate", "grade_generation")
    
    graph.add_conditional_edges(
        "grade_generation",
        route_after_grade_generation,
        {END: END, "rewrite_query": "rewrite_query"},
    )

    return graph.compile()