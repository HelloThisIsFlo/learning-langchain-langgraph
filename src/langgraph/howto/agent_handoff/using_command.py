from langchain_core.tools import tool
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from typing_extensions import Literal

from src.common.utils import get_model, save_graph_png, pretty_print_messages

model = get_model("gpt-4o-mini")


# In this example the agents will be relying on the LLM for doing math.
# In a more realistic follow-up example, we will give the agents tools for
# doing math.

@tool
def transfer_to_multiplication_expert():
    """Ask multiplication agent for help."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to another agent
    # (See the paragraph above)
    return


@tool
def transfer_to_addition_expert():
    """Ask addition agent for help."""
    return


def addition_expert(
        state: MessagesState,
) -> Command[Literal["multiplication_expert", "__end__"]]:
    system_prompt = (
        "You are an addition expert, you can ask the multiplication expert for help with multiplication. "
        "ALWAYS PERFORM ADDITIONS! NEVER MULTIPLICATIONS!"
        "Always do your portion of calculation before the handoff."
    )
    messages = [
                   {"role": "system", "content": system_prompt}
               ] + state["messages"]
    ai_msg = model.bind_tools(
        [transfer_to_multiplication_expert]
    ).invoke(messages)

    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        # NOTE: it's important to insert a tool message here because LLM providers are expecting
        # all AI messages to be followed by a corresponding tool result message
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="multiplication_expert",
            update={"messages": [ai_msg, tool_msg]}
        )

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


def multiplication_expert(
        state: MessagesState,
) -> Command[Literal["addition_expert", "__end__"]]:
    system_prompt = (
        "You are a multiplication expert, you can ask an addition expert for help with addition. "
        "ONLY PERFORM MULTIPLICATIONS! NEVER ADDITION! "
        "Always do your portion of calculation before the handoff."
    )
    messages = [{"role": "system", "content": system_prompt}] + state[
        "messages"]
    ai_msg = model.bind_tools([transfer_to_addition_expert]).invoke(messages)
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="addition_expert",
                       update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}


builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
# we'll always start with the addition expert
builder.add_edge(START, "addition_expert")
graph = builder.compile()

save_graph_png(graph)

for chunk in graph.stream(
        {"messages": [("user", "what's (3 + 5) * 12")]},
):
    pretty_print_messages(chunk)
