"""Command-line entry point for the AI Agent Secretary.

    python cli.py                      # default model from models.toml
    python cli.py --model qwen         # another alias from models.toml
    python cli.py --model openai:<id>  # any provider:model, no models.toml entry needed
    python cli.py --list-models
"""
import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import (
    ACTIVE, DEBUG, RateLimitStop, build_graph, configure_model, list_models, usage_summary,
)


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


def print_models() -> None:
    rows = list_models()
    if not rows:
        print("No models.toml entries found. Use --model provider:model-id (for example groq:<id>).")
        return
    for alias, provider, model, params, is_default in rows:
        extra = f"  {params}" if params else ""
        print(f"{'*' if is_default else ' '} {alias:<14} {provider}:{model}{extra}")
    print("(* = default)")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent Secretary")
    parser.add_argument("--model", help="alias from models.toml, or 'provider:model-id'")
    parser.add_argument("--list-models", action="store_true", help="show configured models and exit")
    args = parser.parse_args()

    if args.list_models:
        print_models()
        return

    try:
        if args.model:
            configure_model(args.model)
        graph = build_graph()
    except Exception as error:  # bad alias, missing package, missing API key, ...
        print(f"Could not set up the model: {error}")
        print("Run 'python cli.py --list-models' to see what is configured.")
        return
    print(f"Model: {ACTIVE['alias']} ({ACTIVE['provider']}:{ACTIVE['model']})")

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

            try:
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
            except RateLimitStop as error:
                print(f"\nStopped: {error}")
                continue

            print(f"\nAgent: {result['messages'][-1].content}")
            if DEBUG:
                print(f"[{usage_summary()}]")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()