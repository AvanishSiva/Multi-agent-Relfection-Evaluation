import argparse
import json
import os
from datetime import datetime

from src.judge import judge_belief_consistency, judge_dialogue_coherence
from src.metrics import (
    compute_avg_rounds_to_deal,
    compute_deal_rate,
    compute_parse_success_rate,
    compute_position_stability,
)
from src.negotiation_state import GAME_INSTANCES, compute_efficiency, compute_fairness_gap
from src.run_game import play_game

CONDITIONS = ["baseline", "reflection", "control"]
INSTANCES = [1, 2, 3]


def run_one(instance_id: int, condition: str, max_rounds: int, repeat: int) -> dict:
    final_state = play_game(instance_id=instance_id, max_rounds=max_rounds, condition=condition)

    a_score, b_score = final_state["a_score"], final_state["b_score"]
    optimal_joint = GAME_INSTANCES[instance_id]["optimal_joint"]
    transcript = final_state["transcript"]
    stability_a = compute_position_stability(transcript, "A")
    stability_b = compute_position_stability(transcript, "B")
    judge_belief_a = judge_belief_consistency(final_state["a_belief_history"], "A")
    judge_belief_b = judge_belief_consistency(final_state["b_belief_history"], "B")
    judge_coherence_a = judge_dialogue_coherence(transcript, "A")
    judge_coherence_b = judge_dialogue_coherence(transcript, "B")

    return {
        "instance": instance_id,
        "condition": condition,
        "repeat": repeat,
        "accepted": final_state["accepted"],
        "round_count": final_state["round_count"],
        "a_score": a_score,
        "b_score": b_score,
        "efficiency": compute_efficiency(a_score, b_score, optimal_joint),
        "fairness_gap": compute_fairness_gap(a_score, b_score),
        "parse_success_rate": compute_parse_success_rate(transcript),
        "position_stability_a": stability_a["stability_score"],
        "position_stability_b": stability_b["stability_score"],
        "contradiction_count_a": stability_a["contradiction_count"],
        "contradiction_count_b": stability_b["contradiction_count"],
        "judge_belief_consistency_a": judge_belief_a["score"],
        "judge_belief_consistency_b": judge_belief_b["score"],
        "judge_coherence_a": judge_coherence_a["score"],
        "judge_coherence_b": judge_coherence_b["score"],
        "judge_reasoning": {
            "belief_a": judge_belief_a["reasoning"],
            "belief_b": judge_belief_b["reasoning"],
            "coherence_a": judge_coherence_a["reasoning"],
            "coherence_b": judge_coherence_b["reasoning"],
        },
        "a_belief_history": final_state["a_belief_history"],
        "b_belief_history": final_state["b_belief_history"],
        "transcript": transcript,
    }


def run_all(instances: list[int], conditions: list[str], repeats: int, max_rounds: int) -> list[dict]:
    results = []
    for instance_id in instances:
        for condition in conditions:
            for repeat in range(repeats):
                print(f"Running instance={instance_id} condition={condition} repeat={repeat + 1}/{repeats} ...")
                results.append(run_one(instance_id, condition, max_rounds, repeat))
    return results


def _avg_non_none(values: list) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize(results: list[dict]) -> dict:
    summary = {}
    for instance_id in sorted({r["instance"] for r in results}):
        for condition in sorted({r["condition"] for r in results}):
            subset = [r for r in results if r["instance"] == instance_id and r["condition"] == condition]
            if not subset:
                continue
            judge_belief_scores = [r.get("judge_belief_consistency_a") for r in subset] + \
                                   [r.get("judge_belief_consistency_b") for r in subset]
            judge_coherence_scores = [r.get("judge_coherence_a") for r in subset] + \
                                      [r.get("judge_coherence_b") for r in subset]
            summary[f"instance_{instance_id}_{condition}"] = {
                "n_runs": len(subset),
                "deal_rate": compute_deal_rate(subset),
                "avg_rounds_to_deal": compute_avg_rounds_to_deal(subset),
                "avg_efficiency": sum(r["efficiency"] for r in subset) / len(subset),
                "avg_fairness_gap": sum(r["fairness_gap"] for r in subset) / len(subset),
                "avg_judge_belief_consistency": _avg_non_none(judge_belief_scores),
                "avg_judge_coherence": _avg_non_none(judge_coherence_scores),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment sweep across conditions and instances.")
    parser.add_argument("--instances", type=int, nargs="+", default=INSTANCES, choices=INSTANCES)
    parser.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_all(args.instances, args.conditions, args.repeats, args.max_rounds)
    summary = summarize(results)

    output_path = args.output
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/experiment_{timestamp}.json"
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"\nSaved {len(results)} runs to {output_path}")
    print("\n=== Summary ===")
    for key, stats in summary.items():
        print(f"{key}: {stats}")


if __name__ == "__main__":
    main()
