from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hr_langgraph_agent import get_agent_reply


app = FastAPI(title="HR Agent API", version="1.0.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        reply = get_agent_reply(user_message)
        return ChatResponse(reply=reply)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
