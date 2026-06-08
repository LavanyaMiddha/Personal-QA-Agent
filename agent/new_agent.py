from typing import Annotated
from unittest import result
from dagster import graph
from sympy import content
from typing_extensions import TypedDict
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from dotenv import load_dotenv

from rag.retriever import Retriever

load_dotenv()

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    confidence_score: float        # ← add these
    confidence_level: str       # 'high', 'medium', 'low' based on retrieval relevance


_retriever = Retriever(search_type="similarity", top_k=5)
# confidence thresholds
HIGH_CONFIDENCE   = 0.7
MEDIUM_CONFIDENCE = 0.4

def _assess_confidence(scores: list[float]) -> tuple[str, float]:
    """
    Returns (confidence_level, avg_score).
    confidence_level: 'high' | 'medium' | 'low'
    """
    if not scores:
        return "low", 0.0
    
    avg_score = sum(scores) / len(scores)
    top_score = max(scores)

    # use top score as primary signal, avg as tiebreaker
    if top_score >= HIGH_CONFIDENCE:
        return "high", avg_score
    elif top_score >= MEDIUM_CONFIDENCE:
        return "medium", avg_score
    else:
        return "low", avg_score


@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from the vector database for a given query."""
    results_with_scores = _retriever.retrieve_with_scores(query)

    if not results_with_scores:
        return "CONFIDENCE:0.00|LEVEL:low|CONTEXT:No relevant context found."

    scores = [score for _, score in results_with_scores]
    confidence_level, avg_score = _assess_confidence(scores)

    # log for debugging
    print(f"Retrieval scores: {[f'{s:.3f}' for s in scores]}")
    print(f"Confidence: {confidence_level} (avg: {avg_score:.3f})")

    # threshold — don't pass low quality context to LLM
    if confidence_level == "low":
        return f"CONFIDENCE:{avg_score:.2f}|LEVEL:low|CONTEXT:No sufficiently relevant context found for this query."

    context_parts = []
    for doc, score in results_with_scores:
        if score >= MEDIUM_CONFIDENCE:  # filter out low scoring chunks
            context_parts.append(
                f"[Score: {score:.3f}] {doc.page_content}\n"
                f"Source: {doc.metadata.get('source', 'unknown')}"
            )

    context = "\n\n".join(context_parts)
    return f"CONFIDENCE:{avg_score:.2f}|LEVEL:{confidence_level}|CONTEXT:{context}"


TOOLS = [retrieve_context]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

# SYSTEM_PROMPT = """You are a helpful assistant that answers questions using retrieved context 
# from a vector database. When a user asks a question, use the retrieve_context tool to fetch 
# relevant information, then provide a clear and accurate answer based on that context. 
# Always cite the source of the information when available."""

SYSTEM_PROMPT_HIGH = """Answer using the provided context. Cite sources when available."""

SYSTEM_PROMPT_MEDIUM = """Answer using the provided context but note that the relevance 
is moderate. If the context seems insufficient, say so clearly. Cite sources when available."""

SYSTEM_PROMPT_LOW = """The retrieved context is not relevant enough to answer confidently.
Say that you don't have sufficient information in your documents to answer this question, 
and suggest the user rephrase or ask something else."""

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
        """LLM decides whether to call a tool or respond directly."""
        query = state["messages"][-1].content

        # retrieve with scores
        raw_context = retrieve_context.invoke({"query": query})

        # parse confidence metadata out of the tool response
        confidence_score = 0.0
        confidence_level = "low"
        context = raw_context

        if raw_context.startswith("CONFIDENCE:"):
            parts = raw_context.split("|", 2)
            confidence_score = float(parts[0].replace("CONFIDENCE:", ""))
            confidence_level = parts[1].replace("LEVEL:", "")
            context = parts[2].replace("CONTEXT:", "") if len(parts) > 2 else ""

        # pick system prompt based on confidence
        if confidence_level == "high":
            system_prompt = SYSTEM_PROMPT_HIGH
        elif confidence_level == "medium":
            system_prompt = SYSTEM_PROMPT_MEDIUM
        else:
            system_prompt = SYSTEM_PROMPT_LOW

        messages = [
            SystemMessage(content=f"{system_prompt}\n\nContext:\n{context}")
        ] + state["messages"]

        response = self.model.invoke(messages)

        # attach confidence metadata to the response for the API to use
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
        }

    def tool_call(self, state: MessagesState):
        """Execute whatever tool the LLM requested and return the result."""
        tool_messages = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_fn = TOOLS_BY_NAME[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )
        return {"messages": tool_messages}


    def should_use_tool(self, state: MessagesState) -> str:
        """Route to tool_call if the LLM requested a tool, otherwise end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_call"
        return END


    def _build_graph(self):
        graph = StateGraph(MessagesState)

        graph.add_node("llm_call", self.llm_call)
        graph.add_node("tool_call", self.tool_call)

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", self.should_use_tool)  # tool_call or END
        graph.add_edge("tool_call", "llm_call")                      # loop back after tool

        return graph.compile(checkpointer=self.checkpointer)


    def invoke(self, query: str, thread_id: str = "default") -> dict:
        result = self.graph.invoke(
            {
                "messages": [{"role": "user", "content": query}],
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
            output_path = "agent_graph.png"
            with open(output_path, "wb") as f:
                f.write(png_data)
            print(f"Graph saved to {output_path}")
        except Exception:
            pass



if __name__ == "__main__":
    agent = Agent()
    #display(agent.display_agent_graph())
    query = "What is DevOps?"
    print(f"Query: {query}\n")
    answer = agent.invoke(query, thread_id="session-1")
    print(f"Answer:\n{answer}")