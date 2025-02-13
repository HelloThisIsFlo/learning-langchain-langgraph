import inspect
import io
import os

from IPython.core.display import Image
from IPython.core.display_functions import display
from PIL import Image as PILImage
from dotenv import load_dotenv
from langchain_core.messages import convert_to_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

MODELS = {
    "local": [
        "qwen2.5-7b-instruct-1m@q8_0",
        "qwen2.5-7b-instruct-1m@q4_k_m",
        "hermes-3-llama-3.2-3b",
        "mistral-small-24b-instruct-2501",
        "phi-4",
        "mistral-nemo-instruct-2407",
        'llama-3.2-3b-instruct',
        'llama-3.2-3b-instruct',
        "granite-3.1-8b-instruct",
        "xlam-7b-r",
        "llama-3-groq-8b-tool-use",
        "bartowski/llama-3.2-3b-instruct",
        "unsloth/llama-3.2-1b-instruct"
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
    ],
    "google": [
        "models/gemini-2.0-flash",
    ],
}


def get_model(model="gpt-4o-mini"):
    load_dotenv()

    temperature = 0.9

    if model == "mistral-nemo-instruct-2407":
        return ChatMistralAI(
            model=model,
            base_url="http://localhost:1234/v1",
            api_key="not-used",
            temperature=temperature
        )

    if model in MODELS["local"]:
        return ChatOpenAI(
            model=model,
            base_url="http://localhost:1234/v1",
            api_key="not-used",
            temperature=temperature
        )
    elif model in MODELS["openai"]:
        # Requires OPENAI_API_KEY
        return ChatOpenAI(model=model, temperature=temperature)
    elif model in MODELS["google"]:
        # Requires GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:
        raise ValueError(f"Invalid model: {model}")


def save_graph_png(graph):
    # Get the filename of the calling module
    caller_frame = inspect.stack()[1]  # [1] to get the direct caller
    caller_filename = inspect.getfile(caller_frame[0])

    # Extract just the filename without the path or extension
    module_name = os.path.splitext(os.path.basename(caller_filename))[0]

    display(Image(graph.get_graph().draw_mermaid_png()))

    # Save the image
    image_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(io.BytesIO(image_data))
    image.save(f"{module_name}__graph.png")


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
