"""Command-line entry point for the AI Agent Secretary."""
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import build_graph


def ask_codebase_path() -> str:
    """Ask for the project directory until it is an existing directory.

    Control characters are dropped: a stray Esc keypress ends up in the string
    as '\\x1b', and used as a path it would create a folder with that name.
    """
    while True:
        raw = input("Codebase path (empty = current directory): ")
        cleaned = "".join(ch for ch in raw if ch.isprintable()).strip()
        path = Path(cleaned or ".").expanduser().resolve()
        if path.is_dir():
            print(f"Using codebase: {path}")
            return str(path)
        print(f"Not an existing directory: {path}. Try again.")


def handle_interrupts(graph, result: dict, config: dict) -> dict:
    """While the graph is paused for approval, ask the user and resume it.

    One question can trigger several approvals in a row (several write_file
    calls, for example), so keep going until the graph stops pausing.
    """
    while "__interrupt__" in result:
        data = result["__interrupt__"][0].value
        print(f"\n[Approval needed] {data['action']}({data['args']})")
        answer = input("Approve? (yes/no): ").strip().lower()
        decision = "approve" if answer in ("yes", "y") else "reject"
        result = graph.invoke(Command(resume=decision), config=config)
    return result


def main() -> None:
    graph = build_graph()
    # The checkpointer keeps the conversation per thread_id (until this process exits).
    config = {"configurable": {"thread_id": "session-1"}}
    codebase_path = ask_codebase_path()

    try:
        while True:
            question = input("\nYou: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "codebase_path": codebase_path,
                    # Counters are reset for every new question; messages accumulate.
                    "step_count": 0,
                    "plan": [],
                    "current_step": 0,
                    "tool_rounds": 0,
                },
                config=config,
            )
            result = handle_interrupts(graph, result, config)
            print(f"\nAgent: {result['messages'][-1].content}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()