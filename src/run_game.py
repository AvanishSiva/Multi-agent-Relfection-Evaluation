import argparse

from src.graph import build_graph
from src.judge import judge_belief_consistency, judge_dialogue_coherence
from src.metrics import compute_parse_success_rate, compute_position_stability
from src.negotiation_state import (
    GAME_INSTANCES,
    compute_efficiency,
    compute_fairness_gap,
    get_initial_state,
)


def play_game(instance_id: int, max_rounds: int, condition: str) -> dict:
    app = build_graph()
    initial_state = get_initial_state(instance_id=instance_id, max_rounds=max_rounds, condition=condition)
    return app.invoke(initial_state, config={"recursion_limit": 50})


def run(instance_id: int, max_rounds: int, condition: str) -> None:
    final_state = play_game(instance_id=instance_id, max_rounds=max_rounds, condition=condition)

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

    print(f"\nA belief history: {final_state['a_belief_history']}")
    print(f"B belief history: {final_state['b_belief_history']}")

    transcript = final_state["transcript"]
    stability_a = compute_position_stability(transcript, "A")
    stability_b = compute_position_stability(transcript, "B")
    print(f"\nParse success rate: {compute_parse_success_rate(transcript):.2f}")
    print(f"Position stability — A: {stability_a['stability_score']:.2f} ({stability_a['contradiction_count']} contradiction(s))")
    print(f"Position stability — B: {stability_b['stability_score']:.2f} ({stability_b['contradiction_count']} contradiction(s))")

    judge_belief_a = judge_belief_consistency(final_state["a_belief_history"], "A")
    judge_belief_b = judge_belief_consistency(final_state["b_belief_history"], "B")
    judge_coherence_a = judge_dialogue_coherence(transcript, "A")
    judge_coherence_b = judge_dialogue_coherence(transcript, "B")
    print(f"\nJudge — belief consistency A: {judge_belief_a['score']}  ({judge_belief_a['reasoning']})")
    print(f"Judge — belief consistency B: {judge_belief_b['score']}  ({judge_belief_b['reasoning']})")
    print(f"Judge — dialogue coherence A: {judge_coherence_a['score']}  ({judge_coherence_a['reasoning']})")
    print(f"Judge — dialogue coherence B: {judge_coherence_b['score']}  ({judge_coherence_b['reasoning']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one negotiation game.")
    parser.add_argument("--instance", type=int, default=1, choices=[1, 2, 3], help="Game instance (1=integrative, 2=mixed, 3=competitive)")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--condition", type=str, default="baseline", choices=["baseline", "reflection", "control"])
    args = parser.parse_args()

    run(instance_id=args.instance, max_rounds=args.max_rounds, condition=args.condition)
