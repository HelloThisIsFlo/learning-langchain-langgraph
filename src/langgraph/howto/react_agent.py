from typing import Literal

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.common.utils import get_model, save_graph_png

model = get_model("gpt-4o-mini")


@tool
def get_weather(city: Literal["nyc", "sf", "porto"]) -> str:
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    elif city == "porto":
        return "It's hot ... but not as hot as Toni"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

graph = create_react_agent(model, tools=tools)


save_graph_png(graph)

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "what is the weather in sf")]}
stream = graph.stream(inputs, stream_mode="values")
print_stream(stream)

print("----------------------------------------------------")
print("----------------------------------------------------")
print("----------------------------------------------------")

# Note: This chat doesn't have memory (didn't set up a Checkpointer)
inputs = {"messages": [("user", "what's the weather in porto?")]}
stream = graph.stream(inputs, stream_mode="values")
print_stream(stream)
