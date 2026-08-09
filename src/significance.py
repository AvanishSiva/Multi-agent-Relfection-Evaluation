import statistics

from scipy.stats import fisher_exact, mannwhitneyu

ALPHA = 0.05


def extract_metric(results: list[dict], condition: str, metric_key: str) -> list[float]:
    return [
        r[metric_key]
        for r in results
        if r["condition"] == condition and r.get(metric_key) is not None
    ]


def mann_whitney_test(sample_a: list[float], sample_b: list[float], label_a: str, label_b: str, metric: str) -> dict:
    result = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    p_value = float(result.pvalue)
    return {
        "metric": metric,
        "test": "mann_whitney_u",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": len(sample_a),
        "n_b": len(sample_b),
        "median_a": statistics.median(sample_a),
        "median_b": statistics.median(sample_b),
        "u_statistic": float(result.statistic),
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


def deal_rate_test(accepted_a: list[bool], accepted_b: list[bool], label_a: str, label_b: str) -> dict:

    a_yes = sum(accepted_a)
    a_no = len(accepted_a) - a_yes
    b_yes = sum(accepted_b)
    b_no = len(accepted_b) - b_yes
    result = fisher_exact([[a_yes, a_no], [b_yes, b_no]])
    p_value = float(result.pvalue)
    return {
        "metric": "deal_rate",
        "test": "fisher_exact",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": len(accepted_a),
        "n_b": len(accepted_b),
        "rate_a": a_yes / len(accepted_a),
        "rate_b": b_yes / len(accepted_b),
        "odds_ratio": float(result.statistic),
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


CONDITION_PAIRS = [("baseline", "reflection"), ("baseline", "control"), ("reflection", "control")]
NUMERIC_METRICS = ["efficiency", "fairness_gap"]


def run_all_comparisons(results: list[dict]) -> list[dict]:
    comparisons = []

    for cond_a, cond_b in CONDITION_PAIRS:
        accepted_a = [r["accepted"] for r in results if r["condition"] == cond_a]
        accepted_b = [r["accepted"] for r in results if r["condition"] == cond_b]
        comparisons.append(deal_rate_test(accepted_a, accepted_b, cond_a, cond_b))

        for metric in NUMERIC_METRICS:
            sample_a = extract_metric(results, cond_a, metric)
            sample_b = extract_metric(results, cond_b, metric)
            comparisons.append(mann_whitney_test(sample_a, sample_b, cond_a, cond_b, metric))

        if "baseline" not in (cond_a, cond_b):
            belief_a = extract_metric(results, cond_a, "judge_belief_consistency_a") + \
                extract_metric(results, cond_a, "judge_belief_consistency_b")
            belief_b = extract_metric(results, cond_b, "judge_belief_consistency_a") + \
                extract_metric(results, cond_b, "judge_belief_consistency_b")
            comparisons.append(mann_whitney_test(belief_a, belief_b, cond_a, cond_b, "judge_belief_consistency"))

    return comparisons
