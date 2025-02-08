import io

from IPython.core.display import Image
from IPython.core.display_functions import display
from PIL import Image as PILImage
from dotenv import load_dotenv
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
