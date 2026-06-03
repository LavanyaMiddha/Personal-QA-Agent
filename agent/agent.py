from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

model = init_chat_model(
    "gemini-3.5-flash",
    model_provider="google-genai",
    temperature=0.4,
    timeout=300,
    max_tokens=25000,
)

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """You are a helpful assistant that can answer questions based on your knowledge."""
query = "What is federated learning? Explain in brief."
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


# agent_result = agent.invoke(
#     {"messages": [{"role": "user", "content": query}]},
#     config={"configurable": {"thread_id": "great-gatsby-lc"}},
# )


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
