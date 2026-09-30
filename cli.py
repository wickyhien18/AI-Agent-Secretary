from agent.graph import build_graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command

graph = build_graph()
config = {"configurable": {"thread_id": "session-1"}}

codebase_path = input("Codebase path: ").strip() or "."

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("exit", "quit"):
        break

    
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)],
         "codebase_path": codebase_path, "step_count": 0, "plan": [], "current_step": 0, "tool_rounds": 0},
        config=config,
    )

    # If the graph paused for approval, keep resuming until it finishes
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        print(f"\n[Approval needed] {interrupt_data['action']}({interrupt_data['args']})")
        answer = input("Approve? (yes/no): ").strip().lower()
        decision = "approve" if answer in ("yes", "y") else "reject"
        result = graph.invoke(Command(resume=decision), config=config)

    print(f"\nAgent: {result['messages'][-1].content}")