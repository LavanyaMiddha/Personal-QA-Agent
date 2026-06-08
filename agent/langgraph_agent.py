from typing import Annotated
from typing_extensions import TypedDict
import operator
import os

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

from rag.retriever import Retriever

load_dotenv()

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    confidence_score: float
    confidence_level: str  # 'high' | 'medium' | 'low'


# --------------------------------------------------------------------------- #
# Retriever (initialized once at module level)
# --------------------------------------------------------------------------- #

_retriever = Retriever(search_type="similarity", top_k=5)

# --------------------------------------------------------------------------- #
# Confidence thresholds
# --------------------------------------------------------------------------- #

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
    else:
        return "low", avg_score


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #

@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from the vector database with confidence scores for a given query."""
    print("Hi, im in retirve context function")
    results_with_scores = _retriever.retrieve_with_scores(query)

    if not results_with_scores:
        return "CONFIDENCE:0.00|LEVEL:low|CONTEXT:No relevant context found."

    scores = [score for _, score in results_with_scores]
    print(f"Retrieval scores: {[f'{s:.3f}' for s in scores]}")  # log scores for debugging
    confidence_level, avg_score = _assess_confidence(scores)

    print(f"Retrieval scores: {[f'{s:.3f}' for s in scores]}")
    print(f"Confidence: {confidence_level} (avg: {avg_score:.3f})")

    context_parts = []
    for doc, score in results_with_scores:
        if score >= MEDIUM_CONFIDENCE:
            context_parts.append(
                f"[Score: {score:.3f}] {doc.page_content}\n"
                f"Source: {doc.metadata.get('source', 'unknown')}"
            )

    context = "\n\n".join(context_parts) if context_parts else "No sufficiently relevant context found."
    return f"CONFIDENCE:{avg_score:.2f}|LEVEL:{confidence_level}|CONTEXT:{context}"


TOOLS = [retrieve_context]

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using retrieved context 
from a vector database. When a user asks a question, use the retrieve_context tool to fetch 
relevant information, then provide a clear and accurate answer based on that context. 
Always cite the source of the information when available."""

# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class Agent:
    def __init__(self):
        self.model = init_chat_model(
            "gemini-3.1-flash-lite",
            model_provider="google-genai",
            temperature=0.4,
            timeout=300,
            max_tokens=25000,
        ).bind_tools(TOOLS)

        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def llm_call(self, state: MessagesState):
        """
        Called on START and after every tool call.
        - On the first call: no tool results yet, LLM will decide to call retrieve_context
        - On subsequent calls: ToolMessage with context is in state, LLM generates final answer
          and parses confidence from the tool output
        """
        messages = state["messages"]

        # parse confidence from the last ToolMessage if present
        confidence_score = state.get("confidence_score", 0.0)
        confidence_level = state.get("confidence_level", "low")
        system_prompt = SYSTEM_PROMPT

        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                raw = msg.content
                if isinstance(raw, str) and raw.startswith("CONFIDENCE:"):
                    parts = raw.split("|", 2)
                    confidence_score = float(parts[0].replace("CONFIDENCE:", ""))
                    confidence_level = parts[1].replace("LEVEL:", "")
                break  # only check the most recent tool message

        print("Selected system prompt based on confidence level:", system_prompt)
        response = self.model.invoke(
            [SystemMessage(content=system_prompt)] + messages
        )

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
        }

    def _build_graph(self):
        graph = StateGraph(MessagesState)

        graph.add_node("llm_call", self.llm_call)
        graph.add_node("tools", ToolNode(TOOLS))  # handles tool execution

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges(
            "llm_call",
            tools_condition,           # routes to "tools" or END
            {"tools", END}
        )
        graph.add_edge("tools", "llm_call")  # loop back after tool

        return graph.compile(checkpointer=self.checkpointer)

    def invoke(self, query: str, thread_id: str = "default") -> dict:
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "llm_calls": 0,
                "confidence_score": 0.0,
                "confidence_level": "low",
            },
            config={"configurable": {"thread_id": thread_id}}
        )

        content = result["messages"][-1].content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return {
            "answer": content,
            "confidence_score": round(result.get("confidence_score", 0.0), 3),
            "confidence_level": result.get("confidence_level", "low"),
        }

    def display_agent_graph(self):
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            with open("agent_graph.png", "wb") as f:
                f.write(png_data)
            print("Graph saved to agent_graph.png")
            import os
            os.startfile("agent_graph.png")
        except Exception as e:
            print(f"Could not save graph: {e}")


if __name__ == "__main__":
    agent = Agent()
    agent.display_agent_graph()
    query = "What is DevOps?"
    print(f"Query: {query}\n")
    answer = agent.invoke(query, thread_id="session-1")
    print(f"Answer:\n{answer['answer']}")
    print(f"Confidence: {answer['confidence_level']} ({answer['confidence_score']})")