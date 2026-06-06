from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from rag.retriever import Retriever
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

model = init_chat_model(
    "gemini-3.5-flash",
    model_provider="google-genai",
    temperature=0.4,
    timeout=300,
    max_tokens=25000,
)

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions based on your knowledge and the retrieved context from a vector database. 
Use the provided context to answer the user's question accurately. 
If the context does not contain relevant information, answer based on your general knowledge. 
Always cite the source of the information from the context when providing an answer."""

query = "What is configuration management? what are the tools used for configuration management?"



# agent_result = agent.invoke(
#     {"messages": [{"role": "user", "content": query}]},
#     config={"configurable": {"thread_id": "great-gatsby-lc"}},
# )

def get_context(query: str) -> str:
    """Tool function to retrieve relevant context from the vector database based on the user's query."""
    retriever = Retriever(search_type="similarity", top_k=5)
    results = retriever.retrieve(query)
    if not results:
        return "No relevant context found."
    context = "\n".join([f"{doc.page_content} (source: {doc.metadata.get('source', 'unknown')})" for doc in results])
    return context

agent = create_agent(
    model=model,
    tools=[get_context],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# Streaming Agent invocation
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": "federated-learning-lc"}},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(f"node: {metadata['langgraph_node']}")
        print(f"content: {token.content_blocks}")
        print("\n")
