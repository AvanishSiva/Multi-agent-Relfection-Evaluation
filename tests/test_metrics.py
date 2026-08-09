import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import (
    compute_position_stability,
    compute_parse_success_rate,
    compute_belief_parse_success_rate,
    compute_deal_rate,
    compute_avg_rounds_to_deal,
)


def _entry(round_, agent, action_type, proposal, parse_ok=True):
    return {
        "round": round_,
        "agent": agent,
        "message": "msg",
        "action_type": action_type,
        "proposal": proposal,
        "parse_ok": parse_ok,
    }


# --- compute_position_stability ---

def test_position_stability_no_proposals_from_agent():
    transcript = [_entry(1, "B", "PROPOSE", {"book": 1, "hat": 1, "ball": 1})]
    result = compute_position_stability(transcript, "A")
    assert result == {"contradiction_count": 0, "opportunities": 0, "stability_score": 1.0}


def test_position_stability_monotonic_sequence_has_no_contradictions():
    transcript = [
        _entry(1, "A", "PROPOSE", {"book": 2, "hat": 2, "ball": 2}),
        _entry(2, "A", "PROPOSE", {"book": 1, "hat": 1, "ball": 1}),
        _entry(3, "A", "PROPOSE", {"book": 0, "hat": 0, "ball": 0}),
    ]
    result = compute_position_stability(transcript, "A")
    assert result["contradiction_count"] == 0
    assert result["stability_score"] == 1.0


def test_position_stability_flip_flop_counts_as_contradiction():
    transcript = [
        _entry(1, "A", "PROPOSE", {"book": 2, "hat": 0, "ball": 0}),
        _entry(2, "A", "PROPOSE", {"book": 0, "hat": 0, "ball": 0}),
        _entry(3, "A", "PROPOSE", {"book": 2, "hat": 0, "ball": 0}),
    ]
    result = compute_position_stability(transcript, "A")
    assert result["contradiction_count"] == 1
    assert result["opportunities"] == 3
    assert round(result["stability_score"], 4) == round(1 - 1 / 3, 4)


def test_position_stability_ignores_opponent_and_accept_entries():
    transcript = [
        _entry(1, "A", "PROPOSE", {"book": 2, "hat": 0, "ball": 0}),
        _entry(1, "B", "PROPOSE", {"book": 0, "hat": 2, "ball": 2}),
        _entry(2, "A", "PROPOSE", {"book": 1, "hat": 0, "ball": 0}),
        _entry(2, "B", "ACCEPT", None),
    ]
    result = compute_position_stability(transcript, "A")
    assert result["opportunities"] == 0
    assert result["contradiction_count"] == 0


# --- compute_parse_success_rate ---

def test_parse_success_rate_all_ok():
    transcript = [_entry(1, "A", "PROPOSE", {"book": 1, "hat": 1, "ball": 1})]
    assert compute_parse_success_rate(transcript) == 1.0


def test_parse_success_rate_mixed():
    transcript = [
        _entry(1, "A", "PROPOSE", {"book": 1, "hat": 1, "ball": 1}, parse_ok=True),
        _entry(2, "B", "PROPOSE", {"book": 1, "hat": 1, "ball": 1}, parse_ok=False),
    ]
    assert compute_parse_success_rate(transcript) == 0.5


def test_parse_success_rate_empty_transcript_defaults_to_one():
    assert compute_parse_success_rate([]) == 1.0


def test_parse_success_rate_filtered_by_agent():
    transcript = [
        _entry(1, "A", "PROPOSE", {"book": 1, "hat": 1, "ball": 1}, parse_ok=False),
        _entry(2, "B", "PROPOSE", {"book": 1, "hat": 1, "ball": 1}, parse_ok=True),
    ]
    assert compute_parse_success_rate(transcript, agent="B") == 1.0


# --- compute_belief_parse_success_rate ---

def test_belief_parse_success_rate_all_ok():
    assert compute_belief_parse_success_rate([True, True, True]) == 1.0


def test_belief_parse_success_rate_mixed():
    assert compute_belief_parse_success_rate([True, False, True, False]) == 0.5


def test_belief_parse_success_rate_empty_defaults_to_one():
    # Matches baseline's condition, which never reflects at all.
    assert compute_belief_parse_success_rate([]) == 1.0


# --- compute_deal_rate / compute_avg_rounds_to_deal ---

def test_deal_rate_empty_runs():
    assert compute_deal_rate([]) == 0.0


def test_deal_rate_mixed():
    runs = [{"accepted": True}, {"accepted": False}, {"accepted": True}, {"accepted": False}]
    assert compute_deal_rate(runs) == 0.5


def test_avg_rounds_to_deal_no_deals_returns_none():
    runs = [{"accepted": False, "round_count": 8}]
    assert compute_avg_rounds_to_deal(runs) is None


def test_avg_rounds_to_deal_averages_only_accepted_runs():
    runs = [
        {"accepted": True, "round_count": 4},
        {"accepted": False, "round_count": 8},
        {"accepted": True, "round_count": 6},
    ]
    assert compute_avg_rounds_to_deal(runs) == 5.0
