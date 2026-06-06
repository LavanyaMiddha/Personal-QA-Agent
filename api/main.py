from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.new_agent import Agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Personal QA Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server (Vite)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# single agent instance shared across requests
agent = Agent()


class QueryRequest(BaseModel):
    query: str
    thread_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    thread_id: str


@app.get("/")
def root():
    return {"status": "Personal QA Agent is running"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        answer = agent.invoke(request.query, thread_id=request.thread_id)
        return QueryResponse(answer=answer, thread_id=request.thread_id)
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}