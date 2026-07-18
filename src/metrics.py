from src.negotiation import ITEMS


def compute_position_stability(transcript: list[dict], agent: str) -> dict:
    # Mechanical proxy for RQ1: counts direction reversals in an agent's own
    # proposed share of each item (e.g. 2 -> 0 -> 2 is a reversal; 2 -> 1 -> 0 is not).
    proposals = [
        entry["proposal"]
        for entry in transcript
        if entry["agent"] == agent and entry["action_type"] == "PROPOSE"
    ]

    contradiction_count = 0
    opportunities = 0
    for item in ITEMS:
        sequence = [p[item] for p in proposals]
        opportunities += max(len(sequence) - 2, 0)
        for i in range(2, len(sequence)):
            diff1 = sequence[i - 1] - sequence[i - 2]
            diff2 = sequence[i] - sequence[i - 1]
            if diff1 != 0 and diff2 != 0 and (diff1 > 0) != (diff2 > 0):
                contradiction_count += 1

    stability_score = 1.0 if opportunities == 0 else 1 - (contradiction_count / opportunities)
    return {
        "contradiction_count": contradiction_count,
        "opportunities": opportunities,
        "stability_score": stability_score,
    }


def compute_parse_success_rate(transcript: list[dict], agent: str | None = None) -> float:
    entries = transcript if agent is None else [e for e in transcript if e["agent"] == agent]
    if not entries:
        return 1.0
    successes = sum(1 for e in entries if e["parse_ok"])
    return successes / len(entries)


def compute_deal_rate(runs: list[dict]) -> float:
    if not runs:
        return 0.0
    deals = sum(1 for r in runs if r["accepted"])
    return deals / len(runs)


def compute_avg_rounds_to_deal(runs: list[dict]) -> float | None:
    rounds = [r["round_count"] for r in runs if r["accepted"]]
    if not rounds:
        return None
    return sum(rounds) / len(rounds)
