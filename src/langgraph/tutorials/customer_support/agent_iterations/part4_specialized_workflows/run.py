import uuid

from langchain_core.messages import ToolMessage

from src.langgraph.tutorials.customer_support.agent_iterations.common import \
    tutorial_questions
from src.langgraph.tutorials.customer_support.agent_iterations.part4_specialized_workflows.graph import \
    part_4_graph
from src.langgraph.tutorials.customer_support.populate_db import update_dates, \
    db
from src.langgraph.tutorials.customer_support.utils import _print_event

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    },
    "recursion_limit": 50,
}


def run_graph(question):
    events = part_4_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values",
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_4_graph.get_state(config)
    while snapshot.next:
        result = None
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            # Just continue
            result = part_4_graph.invoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_4_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0][
                                "id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        _print_event(result, _printed)
        snapshot = part_4_graph.get_state(config)


INTERACTIVE = False

_printed = set()
if INTERACTIVE:
    while True:
        question = input("User: ")
        run_graph(question)
else:
    for question in tutorial_questions:
        run_graph(question)


