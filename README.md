# Langchain & Langgraph

This repo is a place to learn and experiment with Langchain & Langgraph

## How to run?

1. Install `uv`, see [official repo](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
1. Run `uv sync`
1. Create a `.env` file in the root directory and add the following:
   ```bash
   OPENAI_API_KEY=xxxxxxx
   GOOGLE_API_KEY=xxxxxxx
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=xxxxxx
   LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com/  # If using the EU endpoint
   TAVILY_API_KEY=xxxx
   ```
1. Run each example, they are self-contained