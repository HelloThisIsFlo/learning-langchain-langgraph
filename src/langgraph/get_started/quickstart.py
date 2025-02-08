from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()


# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = ChatOpenAI(
    model="qwen2.5-7b-instruct-1m",
    base_url="http://localhost:1234/v1"
)

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

app = create_react_agent(model, tools, checkpointer=checkpointer)

# Use the agent
final_state = app.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config={"configurable": {"thread_id": 42}}
)
resp = final_state["messages"][-1].content
print(resp)

final_state = app.invoke(
    {"messages": [{"role": "user", "content": "what about ny"}]},
    config={"configurable": {"thread_id": 42}}
)
resp = final_state["messages"][-1].content
print(resp)

final_state = app.invoke(
    {"messages": [
        {"role": "user", "content": "what cities did I ask about earlier?"}]},
    config={"configurable": {"thread_id": 42}}
)
resp = final_state["messages"][-1].content
print(resp)
