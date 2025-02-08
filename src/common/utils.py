import io

from IPython.core.display import Image
from IPython.core.display_functions import display
from PIL import Image as PILImage
from dotenv import load_dotenv
from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI


def get_model(local=False):
    if local:
        return ChatOpenAI(
            model="qwen2.5-7b-instruct-1m",
            base_url="http://localhost:1234/v1",
            api_key="not-used",
        )
    else:
        load_dotenv()
        return ChatOpenAI(model="gpt-4o", temperature=0)


def save_graph_png(graph):
    display(Image(graph.get_graph().draw_mermaid_png()))

    # Save the image
    image_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(io.BytesIO(image_data))
    image.save("graph.png")


def stream_graph_updates(graph, user_input: str):
    for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")
