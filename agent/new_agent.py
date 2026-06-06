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


@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from the vector database for a given query."""
    retriever = Retriever(search_type="similarity", top_k=5)
    results = retriever.retrieve(query)
    if not results:
        return "No relevant context found."
    context = "\n\n".join([
        f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}"
        for doc in results
    ])
    return context


TOOLS = [retrieve_context]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using retrieved context 
from a vector database. When a user asks a question, use the retrieve_context tool to fetch 
relevant information, then provide a clear and accurate answer based on that context. 
Always cite the source of the information when available."""

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
        response = self.model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
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


    def invoke(self, query: str, thread_id: str = "default"):
        result = self.graph.invoke(
            {
            "messages": [{"role": "user", "content": query}],
            "llm_calls": 0
            },
        config={"configurable": {"thread_id": thread_id}}
        )
    
        last_message = result["messages"][-1]
        content = last_message.content

        # handle list of content blocks e.g. [{'type': 'text', 'text': '...'}]
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") 
                for block in content 
                if isinstance(block, dict) and block.get("type") == "text"
            )
    
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