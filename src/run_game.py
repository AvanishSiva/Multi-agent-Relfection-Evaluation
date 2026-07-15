import argparse

from src.graph import build_baseline_graph
from src.negotiation_state import (
    GAME_INSTANCES,
    compute_efficiency,
    compute_fairness_gap,
    get_initial_state,
)


def run(instance_id: int, max_rounds: int) -> None:
    app = build_baseline_graph()
    initial_state = get_initial_state(instance_id=instance_id, max_rounds=max_rounds, condition="baseline")
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print(f"=== Instance {instance_id}: {GAME_INSTANCES[instance_id]['description']} ===\n")
    for entry in final_state["transcript"]:
        flag = "" if entry["parse_ok"] else "  (PARSE MISS)"
        print(f"Round {entry['round']} - Agent {entry['agent']}: {entry['message']}")
        print(f"  ACTION: {entry['action_type']} {entry['proposal'] or ''}{flag}\n")

    a_score, b_score = final_state["a_score"], final_state["b_score"]
    optimal_joint = GAME_INSTANCES[instance_id]["optimal_joint"]

    print("=== Result ===")
    print(f"Deal reached: {final_state['accepted']}")
    print(f"Rounds taken: {final_state['round_count']}")
    print(f"Final allocation (Agent A's share): {final_state['final_allocation']}")
    print(f"A score: {a_score}  |  B score: {b_score}")
    print(f"Efficiency: {compute_efficiency(a_score, b_score, optimal_joint):.2f}")
    print(f"Fairness gap: {compute_fairness_gap(a_score, b_score)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one baseline negotiation game.")
    parser.add_argument("--instance", type=int, default=1, choices=[1, 2, 3], help="Game instance (1=integrative, 2=mixed, 3=competitive)")
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()

    run(instance_id=args.instance, max_rounds=args.max_rounds)
