import json
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.graph as graph
import src.judge as judge
from src.run_experiments import run_one, run_all, summarize


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        return _FakeResponse(self._content)


def _with_fake_llm(content):
    graph.llm = _FakeLLM(content)


def _with_fake_judge_llm(content="SCORE: 3\nREASONING: fake judge response for testing."):
    judge.judge_llm = _FakeLLM(content)


# --- run_one (structural, fake LLM, no live Ollama call) ---

def test_run_one_returns_expected_keys_and_types():
    _with_fake_llm("MESSAGE: ok\nACTION: PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}\nBELIEF: some belief text")
    _with_fake_judge_llm()
    result = run_one(instance_id=1, condition="reflection", max_rounds=1, repeat=0)

    expected_keys = {
        "instance", "condition", "repeat", "accepted", "round_count",
        "a_score", "b_score", "efficiency", "fairness_gap", "parse_success_rate",
        "belief_parse_success_rate_a", "belief_parse_success_rate_b",
        "position_stability_a", "position_stability_b",
        "contradiction_count_a", "contradiction_count_b",
        "judge_belief_consistency_a", "judge_belief_consistency_b",
        "judge_coherence_a", "judge_coherence_b", "judge_reasoning",
        "a_belief_history", "b_belief_history", "transcript",
    }
    assert expected_keys.issubset(result.keys())
    assert result["instance"] == 1
    assert result["condition"] == "reflection"
    assert result["repeat"] == 0
    assert result["round_count"] == 1
    assert result["accepted"] is False
    assert result["a_score"] == 0 and result["b_score"] == 0
    assert result["efficiency"] == 0.0
    assert result["a_belief_history"] == ["some belief text"]
    assert result["b_belief_history"] == ["some belief text"]
    assert result["belief_parse_success_rate_a"] == 1.0
    assert result["belief_parse_success_rate_b"] == 1.0
    assert len(result["transcript"]) == 2
    assert result["judge_belief_consistency_a"] is None
    assert result["judge_coherence_a"] == 3


def test_run_one_baseline_has_no_belief_history():
    _with_fake_llm("MESSAGE: ok\nACTION: PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}")
    _with_fake_judge_llm()
    result = run_one(instance_id=2, condition="baseline", max_rounds=1, repeat=0)
    assert result["a_belief_history"] == []
    assert result["b_belief_history"] == []
    assert result["judge_belief_consistency_a"] is None
    assert result["belief_parse_success_rate_a"] == 1.0
    assert result["belief_parse_success_rate_b"] == 1.0


# --- summarize (pure, no LLM) ---

def _fake_result(instance, condition, accepted, round_count, efficiency, fairness_gap):
    return {
        "instance": instance, "condition": condition,
        "accepted": accepted, "round_count": round_count,
        "efficiency": efficiency, "fairness_gap": fairness_gap,
        "belief_parse_success_rate_a": 1.0, "belief_parse_success_rate_b": 1.0,
    }


def test_summarize_groups_by_instance_and_condition():
    results = [
        _fake_result(1, "baseline", True, 4, 1.0, 0),
        _fake_result(1, "baseline", False, 8, 0.0, 0),
        _fake_result(1, "reflection", True, 3, 0.9, 2),
    ]
    summary = summarize(results)

    assert set(summary.keys()) == {"instance_1_baseline", "instance_1_reflection"}
    baseline_stats = summary["instance_1_baseline"]
    assert baseline_stats["n_runs"] == 2
    assert baseline_stats["deal_rate"] == 0.5
    assert baseline_stats["avg_rounds_to_deal"] == 4
    assert baseline_stats["avg_efficiency"] == 0.5

    reflection_stats = summary["instance_1_reflection"]
    assert reflection_stats["n_runs"] == 1
    assert reflection_stats["deal_rate"] == 1.0
    assert reflection_stats["avg_rounds_to_deal"] == 3


def test_summarize_empty_results():
    assert summarize([]) == {}


# --- run_all checkpointing / resume ---

def test_run_all_writes_a_checkpoint_after_every_game():
    _with_fake_llm("MESSAGE: ok\nACTION: PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}")
    _with_fake_judge_llm()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = os.path.join(tmp, "checkpoint.json")
        run_all(instances=[1], conditions=["baseline"], repeats=2, max_rounds=1, checkpoint_path=checkpoint_path)

        with open(checkpoint_path) as f:
            saved = json.load(f)
        assert len(saved["results"]) == 2


def test_run_all_resumes_and_skips_already_completed_games():
    _with_fake_llm("MESSAGE: ok\nACTION: PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}")
    _with_fake_judge_llm()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = os.path.join(tmp, "checkpoint.json")
        prior = [run_one(instance_id=1, condition="baseline", max_rounds=1, repeat=0)]
        with open(checkpoint_path, "w") as f:
            json.dump({"results": prior}, f)

        results = run_all(instances=[1], conditions=["baseline"], repeats=2, max_rounds=1, checkpoint_path=checkpoint_path)

        assert len(results) == 2
        repeats_present = sorted(r["repeat"] for r in results)
        assert repeats_present == [0, 1]


def test_run_all_without_checkpoint_path_behaves_as_before():
    _with_fake_llm("MESSAGE: ok\nACTION: PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}")
    _with_fake_judge_llm()
    results = run_all(instances=[1], conditions=["baseline"], repeats=1, max_rounds=1)
    assert len(results) == 1
