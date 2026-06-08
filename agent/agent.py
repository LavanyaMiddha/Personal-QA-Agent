from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from rag.retriever import Retriever
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

load_dotenv()

class RetrievedDocument(BaseModel):
    source: str = Field(..., description="The file path source of the retrieved document.")
    page_number: int = Field(..., description="The page number of the retrieved document.")
    score: float = Field(..., description="The similarity score of the retrieved document.")
    page_content: str = Field(..., description="The content of the retrieved document.")

class AgentResponse(BaseModel):
    answer: str = Field(..., description="The answer to the user's question based on the retrieved context.")
    confidence_score: float = Field(..., description="The average confidence score of the retrieved documents.")
    confidence_level: Literal["high", "medium", "low"] = Field(..., description="The confidence level based on the scores of the retrieved documents.")
    retrieved_documents: list[RetrievedDocument] = Field(default=[], description="All retrieved documents. Include every single document returned by retrieve_context, not just the most relevant one.")

_retriever = Retriever(search_type="similarity", top_k=3)
HIGH_CONFIDENCE   = 0.8
MEDIUM_CONFIDENCE = 0.3

   
def _assess_confidence(scores: list[float]) -> tuple[str, float]:
    if not scores:
        return "low", 0.0
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    if top_score >= HIGH_CONFIDENCE:
        return "high", avg_score
    elif top_score >= MEDIUM_CONFIDENCE:
        return "medium", avg_score
    else:
        return "low", avg_score

@tool("retrieve_context", description="Retrieves relevant context from the knowledge base for a given query.")
def retriever_context(query:str)->dict:
    results = _retriever.retrieve_with_scores(query)
    if not results:
        return {
            "documents": [],
            "scores": [],
            "avg_confidence_score": 0.0,
            "avg_confidence_level": "low"
        }
    documents, scores = [doc for doc, _ in results], [score for _, score in results]
    confidence_level, avg_confidence_score = _assess_confidence(scores)
    serialized_documents = [
        {
            "page_content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),  # ← explicit metadata field
            "page_number": doc.metadata.get("page_number", 0),
        }
        for doc in documents
    ]
    # print(f"Serialized documents: {serialized_documents}")
    return {
        "documents": serialized_documents,
        "scores": scores,
        "avg_confidence_score": avg_confidence_score,
        "avg_confidence_level": confidence_level
    }


TOOLS = [retriever_context]


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using retrieved context 
from a vector database. When a user asks a question, use the retrieve_context tool to fetch 
relevant information, then provide a clear and accurate answer based on that context.
Always cite all the sources and their confidence scores for the information when available."""


class Agent:
    def __init__(self):
        self.model = init_chat_model(
            "gemini-3.1-flash-lite",
            model_provider="google-genai",
            temperature=0.4,
            timeout=300,
            max_tokens=2500,
        ).bind_tools(TOOLS)

        self.checkpointer = InMemorySaver()
        set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

        self.agent = create_agent(
            model=self.model,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            response_format=AgentResponse,
            middleware=[
                SummarizationMiddleware(
                    model="google_genai:gemini-3.1-flash-lite",
                    trigger=[("tokens", 3000), ("messages", 6)],
                    keep=("messages", 2),
                ),
                ModelCallLimitMiddleware(
                    thread_limit=20,
                    run_limit=5,
                    exit_behavior="end",
                ),
                ToolCallLimitMiddleware(
                    tool_name="retrieve_context",
                    thread_limit=10,
                    run_limit=5,
                ),
                ToolRetryMiddleware(
                    max_retries=3,
                ),
            ]
        )

    def invoke(self, query: str, thread_id: str = "default") -> str:

        result = self.agent.invoke( {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        print("Raw agent output:", result)
        structured = result.get("structured_response")
        if structured and isinstance(structured, AgentResponse):
            return structured
        else:
            return AgentResponse(
                answer=result.get("response", "Sorry, I couldn't generate an answer."),
                confidence_score=0.0,
                confidence_level="low"
            )
        

if __name__ == "__main__":
    agent = Agent()

    query = "What is config management? What are the tools used for configuration management?"
    print(f"Query 1: {query}\n")

    response = agent.invoke(query, thread_id="session-1")
    print(f"Answer:\n{response.answer}")
    query = "What is config management? What are the tools used for configuration management?"
    print("\n\n\n")
    print(f"Query 2: {query}\n")

    response = agent.invoke(query, thread_id="session-1")
    print(f"Answer:\n{response.answer}")
    #print(f"Confidence: {response.confidence_level} ({response.confidence_score})")
        
    