import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.significance import (
    extract_metric,
    mann_whitney_test,
    deal_rate_test,
    run_all_comparisons,
    CONDITION_PAIRS,
)


def _result(condition, accepted=True, efficiency=0.5, fairness_gap=1, belief_a=None, belief_b=None):
    return {
        "condition": condition,
        "accepted": accepted,
        "efficiency": efficiency,
        "fairness_gap": fairness_gap,
        "judge_belief_consistency_a": belief_a,
        "judge_belief_consistency_b": belief_b,
    }


# --- extract_metric ---

def test_extract_metric_filters_by_condition():
    results = [_result("baseline", efficiency=0.1), _result("reflection", efficiency=0.9)]
    assert extract_metric(results, "reflection", "efficiency") == [0.9]


def test_extract_metric_skips_none_values():
    results = [
        _result("baseline", belief_a=None),
        _result("baseline", belief_a=4),
    ]
    assert extract_metric(results, "baseline", "judge_belief_consistency_a") == [4]


# --- mann_whitney_test ---

def test_mann_whitney_test_returns_expected_shape():
    result = mann_whitney_test([1, 2, 3], [4, 5, 6], "baseline", "reflection", "efficiency")
    assert result["metric"] == "efficiency"
    assert result["test"] == "mann_whitney_u"
    assert result["group_a"] == "baseline"
    assert result["group_b"] == "reflection"
    assert result["n_a"] == 3
    assert result["n_b"] == 3
    assert result["median_a"] == 2
    assert result["median_b"] == 5
    assert isinstance(result["p_value"], float)
    assert isinstance(result["significant"], bool)


def test_mann_whitney_test_detects_a_clear_difference_as_significant():
    low = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11]
    high = [0.9, 0.85, 0.95, 0.88, 0.92, 0.91]
    result = mann_whitney_test(low, high, "control", "reflection", "efficiency")
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_mann_whitney_test_detects_identical_distributions_as_not_significant():
    same_a = [1, 2, 3, 4, 5]
    same_b = [1, 2, 3, 4, 5]
    result = mann_whitney_test(same_a, same_b, "baseline", "control", "fairness_gap")
    assert result["significant"] is False


# --- deal_rate_test ---

def test_deal_rate_test_returns_expected_shape():
    result = deal_rate_test([True, True, False], [True, True, True], "baseline", "reflection")
    assert result["metric"] == "deal_rate"
    assert result["test"] == "fisher_exact"
    assert result["n_a"] == 3
    assert result["n_b"] == 3
    assert result["rate_a"] == 2 / 3
    assert result["rate_b"] == 1.0
    assert isinstance(result["p_value"], float)


def test_deal_rate_test_identical_rates_not_significant():
    result = deal_rate_test([True, False, True, False], [True, False, False, True], "baseline", "control")
    assert result["significant"] is False


# --- run_all_comparisons ---

def test_run_all_comparisons_covers_every_condition_pair():
    results = [
        _result("baseline", belief_a=None),
        _result("reflection", belief_a=4, belief_b=4),
        _result("control", belief_a=3, belief_b=3),
    ] * 5  # repeat to give each group n=5, enough for the test to run without error

    comparisons = run_all_comparisons(results)
    pairs_seen = {(c["group_a"], c["group_b"]) for c in comparisons}
    assert pairs_seen == set(CONDITION_PAIRS)


def test_run_all_comparisons_skips_belief_consistency_when_baseline_involved():
    results = [
        _result("baseline", belief_a=None),
        _result("reflection", belief_a=4, belief_b=4),
        _result("control", belief_a=3, belief_b=3),
    ] * 5

    comparisons = run_all_comparisons(results)
    belief_comparisons = [c for c in comparisons if c["metric"] == "judge_belief_consistency"]
    assert len(belief_comparisons) == 1
    assert {belief_comparisons[0]["group_a"], belief_comparisons[0]["group_b"]} == {"reflection", "control"}


def test_run_all_comparisons_includes_deal_rate_and_numeric_metrics_per_pair():
    results = [
        _result("baseline", belief_a=None),
        _result("reflection", belief_a=4, belief_b=4),
        _result("control", belief_a=3, belief_b=3),
    ] * 5

    comparisons = run_all_comparisons(results)
    metrics_for_baseline_reflection = {
        c["metric"] for c in comparisons
        if {c["group_a"], c["group_b"]} == {"baseline", "reflection"}
    }
    assert metrics_for_baseline_reflection == {"deal_rate", "efficiency", "fairness_gap"}
