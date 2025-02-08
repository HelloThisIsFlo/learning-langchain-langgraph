import getpass
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise "Please set the OPENAI_API_KEY environment variable"

    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    resp = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    resp = model.invoke([HumanMessage(content="What's my name?")])

    resp = model.invoke(
        [
            HumanMessage(content="Hi! I'm Bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
            HumanMessage(content="What's my name?"),
        ]
    )

    # Keeping State with LangGraph ####
    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the function that calls the model
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    def new_config(thread_id):
        return {
            "configurable": {
                # The thread_id is a unique identifier for the conversation
                "thread_id": thread_id
            }
        }

    config = new_config('abc123')

    query = "Hi! I'm Bob."
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

    config = new_config('different_thread')
    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

    config = new_config('abc123')  # Back to the original thread
    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()


    # TODO: I stopped at 'Prompt Templates'
    #  Continue there => https://python.langchain.com/docs/tutorials/chatbot/#prompt-templates


if __name__ == "__main__":
    main()
