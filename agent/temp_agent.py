import sys
sys.path.insert(0, r"C:\Personal-QA-Agent")

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

from rag.retriever import Retriever

load_dotenv()

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class AgentState(MessagesState):
    llm_calls: int = 0
    confidence_score: float = 0.0
    confidence_level: str = "low"


# --------------------------------------------------------------------------- #
# Structured response format
# --------------------------------------------------------------------------- #

class AgentResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The answer to the user's question based on the retrieved context."
    )
    confidence_score: float = Field(
        ...,
        description="The average confidence score of the retrieved documents."
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        ...,
        description="The confidence level based on the scores of the retrieved documents."
    )


# --------------------------------------------------------------------------- #
# Retriever singleton
# --------------------------------------------------------------------------- #

_retriever = Retriever(search_type="similarity", top_k=3)

HIGH_CONFIDENCE   = 0.8
MEDIUM_CONFIDENCE = 0.5


def _assess_confidence(scores: list[float]) -> tuple[str, float]:
    if not scores:
        return "low", 0.0
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    if top_score >= HIGH_CONFIDENCE:
        return "high", avg_score
    elif top_score >= MEDIUM_CONFIDENCE:
        return "medium", avg_score
    return "low", avg_score


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #

@tool(
    "retrieve_context",
    response_format="content_and_artifact",
    description="Retrieves relevant context from the knowledge base for a given query."
)
def retrieve_context(query: str) -> tuple[str, dict]:
    results = _retriever.retrieve_with_scores(query)

    if not results:
        return "No relevant context found.", {
            "documents": [],
            "confidence_score": 0.0,
            "confidence_level": "low",
        }

    documents = [doc for doc, _ in results]
    scores    = [score for _, score in results]

    print(f"Raw scores: {[f'{s:.3f}' for s in scores]}")

    confidence_level, avg_score = _assess_confidence(scores)
    print(f"Confidence: {confidence_level} (avg: {avg_score:.3f})")

    # content — readable string for the LLM
    content = "\n\n".join(
        f"[Score: {score:.3f}] {doc.page_content}\n"
        f"Source: {doc.metadata.get('source', 'unknown')}"
        for doc, score in results
        if score >= MEDIUM_CONFIDENCE
    ) or "No sufficiently relevant context found."

    # artifact — structured data for your code
    artifact = {
        "documents": documents,
        "scores": scores,
        "confidence_score": avg_score,
        "confidence_level": confidence_level,
    }

    return content, artifact


TOOLS = [retrieve_context]

# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using retrieved context 
from a vector database. When a user asks a question, use the retrieve_context tool to fetch 
relevant information, then provide a clear and accurate answer based on that context.
Consider the confidence score and level when using the retrieved context. 
If the confidence is low, be cautious and ask the user for clarification.
Always cite the source of the information when available."""


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class Agent:
    def __init__(self):
        self.model = init_chat_model(
            "gemini-2.0-flash",
            model_provider="google-genai",
            temperature=0.4,
            timeout=300,
            max_tokens=25000,
        )
        # note: no .bind_tools() — create_agent handles this

        self.agent = create_agent(
            model=self.model,
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
            response_format=AgentResponse,
            middleware=[
                SummarizationMiddleware(
                    model="google-genai:gemini-2.0-flash",
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

    def invoke(self, query: str, thread_id: str = "default") -> AgentResponse:
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )

        # extract structured response
        structured = result.get("structured_response")
        if structured and isinstance(structured, AgentResponse):
            return structured

        # fallback — parse confidence from artifact if structured response missing
        confidence_score = 0.0
        confidence_level = "low"
        for msg in reversed(result["messages"]):
            if hasattr(msg, "artifact") and isinstance(msg.artifact, dict):
                confidence_score = msg.artifact.get("confidence_score", 0.0)
                confidence_level = msg.artifact.get("confidence_level", "low")
                break

        content = result["messages"][-1].content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return AgentResponse(
            answer=content,
            confidence_score=round(confidence_score, 3),
            confidence_level=confidence_level,
        )

    def display_agent_graph(self):
        try:
            import os
            png_data = self.agent.get_graph().draw_mermaid_png()
            with open("agent_graph.png", "wb") as f:
                f.write(png_data)
            print("Graph saved to agent_graph.png")
            os.startfile("agent_graph.png")
        except Exception as e:
            print(f"Could not save graph: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    agent = Agent()
    agent.display_agent_graph()

    query = "What is DevOps?"
    print(f"Query: {query}\n")

    response = agent.invoke(query, thread_id="session-1")
    print(f"Answer:\n{response.answer}")
    print(f"Confidence: {response.confidence_level} ({response.confidence_score})")