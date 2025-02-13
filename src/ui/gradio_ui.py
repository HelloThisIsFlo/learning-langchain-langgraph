import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

from src.common.utils import get_model

model = get_model("gpt-4o-mini")


def invoke_chatbot(state: MessagesState):
    resp = model.invoke(state["messages"])
    return {"messages": resp}


graph_builder = StateGraph(state_schema=MessagesState)

graph_builder.add_node("chatbot", invoke_chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def chatbot_response(message, _history):
    config = {
        "configurable": {
            "thread_id": "abc123"
        }
    }
    resp = graph.invoke({"messages": [HumanMessage(message)]}, config)
    return resp["messages"][-1].content


demo = gr.ChatInterface(
    fn=chatbot_response,
    type="messages"
)

if __name__ == '__main__':
    demo.launch()
