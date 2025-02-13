from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from src.common.utils import get_model
from src.langgraph.tutorials.customer_support.agent_iterations.part4_specialized_workflows.state import \
    State

# llm = get_model("models/gemini-2.0-flash")
# llm = get_model("gpt-4o-mini")
llm = get_model("gpt-4o")
# llm = get_model("bartowski/llama-3.2-3b-instruct")

# llm = get_model("unsloth/llama-3.2-1b-instruct")
# llm = get_model("mistral-nemo-instruct-2407")
# llm = get_model("o3-mini")


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
            ):
                messages = state["messages"] + [
                    ("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }
