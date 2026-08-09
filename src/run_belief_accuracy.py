
import argparse
import json

from src.judge import judge_belief_accuracy
from src.negotiation_state import GAME_INSTANCES


def score_game(game: dict) -> dict:
    instance = GAME_INSTANCES[game["instance"]]
    a_accuracy = judge_belief_accuracy(game["a_belief_history"], instance["b_values"], "A", "B")
    b_accuracy = judge_belief_accuracy(game["b_belief_history"], instance["a_values"], "B", "A")
    return {
        "instance": game["instance"],
        "condition": game["condition"],
        "repeat": game["repeat"],
        "belief_accuracy_a": a_accuracy["score"],
        "belief_accuracy_b": b_accuracy["score"],
        "reasoning_a": a_accuracy["reasoning"],
        "reasoning_b": b_accuracy["reasoning"],
    }


def summarize(scored: list[dict]) -> dict:
    summary = {}
    for instance_id in sorted({s["instance"] for s in scored}):
        for condition in sorted({s["condition"] for s in scored}):
            subset = [s for s in scored if s["instance"] == instance_id and s["condition"] == condition]
            if not subset:
                continue
            scores = [s["belief_accuracy_a"] for s in subset if s["belief_accuracy_a"] is not None] + \
                     [s["belief_accuracy_b"] for s in subset if s["belief_accuracy_b"] is not None]
            summary[f"instance_{instance_id}_{condition}"] = {
                "n_scored": len(scores),
                "avg_belief_accuracy": sum(scores) / len(scores) if scores else None,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Score belief accuracy against ground truth for an existing sweep.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    scored = []
    total = len(data["results"])
    for i, game in enumerate(data["results"], start=1):
        print(f"Scoring game {i}/{total} (instance={game['instance']} condition={game['condition']} repeat={game['repeat']}) ...")
        scored.append(score_game(game))
        with open(args.output, "w") as f:
            json.dump({"results": scored, "summary": summarize(scored)}, f, indent=2)

    print(f"\nSaved {len(scored)} scored games to {args.output}")
    print("\n=== Summary ===")
    for key, stats in summarize(scored).items():
        print(f"{key}: {stats}")


if __name__ == "__main__":
    main()
