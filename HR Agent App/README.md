# HR Management LangGraph Agent

This folder contains a LangGraph-based HR assistant that supports:

- Employee detail lookup
- Leave balance checks
- Leave request submission
- HR policy lookup
- REST API via FastAPI
- Web chat UI via Streamlit

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Export your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

4. Run the agent:

```bash
python hr_langgraph_agent.py
```

## Run API

```bash
uvicorn api:app --reload --port 8000
```

Test endpoints:

- Health: `GET http://127.0.0.1:8000/health`
- Chat: `POST http://127.0.0.1:8000/chat`

Example payload:

```json
{
  "message": "Check leave balance for E101"
}
```

## Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Example prompts

- `Get employee details for E101`
- `How many leaves are available for E102?`
- `Submit leave request for E101 for 3 days due to family event`
- `What is the leave policy?`
