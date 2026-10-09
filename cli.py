import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import DEBUG, RateLimitStop, build_graph, list_models, usage_summary, use_model


def ask_codebase_path() -> str:
    while True:
        raw = input("Codebase path (empty = current directory): ")
        cleaned = "".join(ch for ch in raw if ch.isprintable()).strip()
        path = Path(cleaned or ".").expanduser().resolve()
        if path.is_dir():
            print(f"Using codebase: {path}")
            return str(path)
        print(f"Not an existing directory: {path}. Try again.")


def handle_interrupts(graph, result: dict, config: dict) -> dict:
    while "__interrupt__" in result:
        data = result["__interrupt__"][0].value
        print(f"\n[Approval needed] {data['action']}({data['args']})")
        answer = input("Approve? (yes/no): ").strip().lower()
        decision = "approve" if answer in ("yes", "y") else "reject"
        result = graph.invoke(Command(resume=decision), config=config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent Secretary")
    parser.add_argument("--model", help="model name from models.toml")
    parser.add_argument("--list-models", action="store_true", help="show the models and exit")
    args = parser.parse_args()

    if args.list_models:
        names, default = list_models()
        print("\n".join(f"{'*' if name == default else ' '} {name}" for name in names))
        return

    try:
        alias = use_model(args.model)
        graph = build_graph()
    except Exception as error:
        print(f"Could not set up the model: {error}")
        return
    print(f"Model: {alias}")

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
                        "step_count": 0,
                        "plan": [],
                        "current_step": 0,
                        "tool_rounds": 0,
                        "partial_reads": {},
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