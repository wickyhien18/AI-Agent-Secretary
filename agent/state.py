from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Shared state passed between every node in the graph.
    Each node reads part of this state and returns a partial update;
    LangGraph merges the update back into the full state automatically.
    """

    #Full converstation history, include AIMessage (with tool_calls)
    #and ToolMessage (tool_results). 'add_message' is a reducer:
    #instead of overwriting this list on every node return, it appends
    #new messages to the existing ones,
    messages: Annotated[list, add_messages]

    # Path to the codebase currently being inspected.
    # None when the current question does't involve code at all
    codebase_path = str | None

    # Number of plan -> act -> observe cycles completed so far.
    # Used to hard-stop the loop if the agent keep looping
    # without reaching an answer (avoids infinite cost/latency)
    step_count: int