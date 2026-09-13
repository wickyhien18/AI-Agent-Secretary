from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage

from state import AgentState
from tools import tools, tools_by_name

from config import LLM_MODEL

MAX_STEPS = 5

SYSTEM_PROMPT = """You are a coding assistant and research agent.
If a required tool parameter is missing from the user's question
(for example, no file path or no search topic), ask the user to
clarify BEFORE calling the tool. Do not guess or invent parameter
values."""

llm = ChatGroq(model=LLM_MODEL)
llm_with_tools = llm.bind_tools(tools)

def plan(state: AgentState) -> dict:
    """LLM reads the conversation so far and decides: answer directly,
    or request a tool call."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_plan(state: AgentState) -> str:
    """Conditional edge: check the last AIMessage for tool_calls."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "act"
    return END


def act(state: AgentState) -> dict:
    """Execute every tool call requested by the last AIMessage."""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = dict(tool_call["args"])

        # Defense-in-depth validation (Bug 2 fix): never trust a
        # model-guessed codebase_path — always use the one already
        # known in state, or fail clearly instead of guessing.
        if name == "search_codebase":
            if state.get("codebase_path"):
                args["codebase_path"] = state["codebase_path"]
            else:
                tool_messages.append(ToolMessage(
                    content="Missing codebase_path. Ask the user which repository to search.",
                    tool_call_id=tool_call["id"],
                ))
                continue

        if not args.get("query", "").strip():
            tool_messages.append(ToolMessage(
                content="Missing required 'query' parameter.",
                tool_call_id=tool_call["id"],
            ))
            continue

        result = tools_by_name[name].invoke(args)
        tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    return {"messages": tool_messages}


def observe(state: AgentState) -> dict:
    """Bookkeeping node: increment the step counter after each act."""
    return {"step_count": state["step_count"] + 1}


def route_after_observe(state: AgentState) -> str:
    """Conditional edge: loop back to plan, or force stop if too many steps."""
    if state["step_count"] >= MAX_STEPS:
        return END
    return "plan"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("observe", observe)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_after_plan, {"act": "act", END: END})
    graph.add_edge("act", "observe")
    graph.add_conditional_edges("observe", route_after_observe, {"plan": "plan", END: END})

    return graph.compile()