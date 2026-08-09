import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.judge as judge
from src.judge import (
    parse_judge_output,
    build_belief_consistency_judge_prompt,
    build_coherence_judge_prompt,
    build_belief_accuracy_judge_prompt,
    judge_belief_consistency,
    judge_dialogue_coherence,
    judge_belief_accuracy,
)


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _FakeResponse(self._content)


class _RaisingLLM:
    def invoke(self, prompt):
        raise AssertionError("judge_llm should not have been called")


def _with_fake_judge_llm(content):
    fake = _FakeLLM(content)
    judge.judge_llm = fake
    return fake


# --- parse_judge_output ---

def test_parse_judge_output_well_formed():
    output = "SCORE: 4\nREASONING: Mostly consistent, one minor revision in belief 3."
    result = parse_judge_output(output)
    assert result["parse_ok"] is True
    assert result["score"] == 4
    assert result["reasoning"] == "Mostly consistent, one minor revision in belief 3."


def test_parse_judge_output_missing_score_falls_back():
    output = "This response forgot the required format."
    result = parse_judge_output(output)
    assert result["parse_ok"] is False
    assert result["score"] is None


def test_parse_judge_output_score_out_of_range_falls_back():
    output = "SCORE: 9\nREASONING: way off scale"
    result = parse_judge_output(output)
    assert result["parse_ok"] is False
    assert result["score"] is None


def test_parse_judge_output_case_insensitive_and_whitespace():
    output = "score:   3   \nreasoning:   Some back and forth but mostly fine.   "
    result = parse_judge_output(output)
    assert result["parse_ok"] is True
    assert result["score"] == 3
    assert result["reasoning"] == "Some back and forth but mostly fine."


# --- prompt builders (structural, no LLM call) ---

def test_belief_consistency_prompt_numbers_each_entry():
    prompt = build_belief_consistency_judge_prompt(["B likes balls.", "B likes hats now."], "A")
    assert "1. B likes balls." in prompt
    assert "2. B likes hats now." in prompt
    assert "SCORE:" in prompt


def test_coherence_prompt_includes_transcript_turns():
    transcript = [
        {"agent": "A", "message": "opening offer", "action_type": "PROPOSE", "proposal": {"book": 1, "hat": 0, "ball": 0}},
        {"agent": "B", "message": "counter", "action_type": "ACCEPT", "proposal": None},
    ]
    prompt = build_coherence_judge_prompt(transcript, "B")
    assert "opening offer" in prompt
    assert "counter" in prompt
    assert "ACCEPT" in prompt
    assert "SCORE:" in prompt


# --- judge_belief_consistency / judge_dialogue_coherence ---

def test_judge_belief_consistency_short_circuits_on_insufficient_history():
    judge.judge_llm = _RaisingLLM()
    result = judge_belief_consistency([], "A")
    assert result["score"] is None
    assert result["parse_ok"] is True

    result = judge_belief_consistency(["only one entry"], "A")
    assert result["score"] is None


def test_judge_belief_consistency_calls_llm_and_parses_score():
    fake = _with_fake_judge_llm("SCORE: 5\nREASONING: Beliefs only sharpened over time, no contradictions.")
    result = judge_belief_consistency(["B likes balls.", "B really likes balls, confirmed."], "A")
    assert fake.calls == 1
    assert result["score"] == 5
    assert result["parse_ok"] is True


def test_judge_dialogue_coherence_short_circuits_when_agent_has_no_turns():
    judge.judge_llm = _RaisingLLM()
    transcript = [{"agent": "B", "message": "hi", "action_type": "PROPOSE", "proposal": {"book": 1, "hat": 0, "ball": 0}}]
    result = judge_dialogue_coherence(transcript, "A")
    assert result["score"] is None
    assert result["parse_ok"] is True


def test_judge_dialogue_coherence_calls_llm_and_parses_score():
    fake = _with_fake_judge_llm("SCORE: 2\nREASONING: Agent B's message claimed compromise but the offer never changed.")
    transcript = [
        {"agent": "A", "message": "opening", "action_type": "PROPOSE", "proposal": {"book": 1, "hat": 0, "ball": 0}},
        {"agent": "B", "message": "I'll compromise", "action_type": "PROPOSE", "proposal": {"book": 1, "hat": 0, "ball": 0}},
    ]
    result = judge_dialogue_coherence(transcript, "B")
    assert fake.calls == 1
    assert result["score"] == 2


# --- build_belief_accuracy_judge_prompt / judge_belief_accuracy ---

def test_belief_accuracy_prompt_includes_true_values_and_belief():
    prompt = build_belief_accuracy_judge_prompt(
        "Agent B seems to value hats most.", {"book": 0, "hat": 1, "ball": 4}, "A", "B"
    )
    assert "book: 0 points each" in prompt
    assert "hat: 1 points each" in prompt
    assert "ball: 4 points each" in prompt
    assert "Agent B seems to value hats most." in prompt
    assert "SCORE:" in prompt


def test_judge_belief_accuracy_short_circuits_on_empty_history():
    judge.judge_llm = _RaisingLLM()
    result = judge_belief_accuracy([], {"book": 4, "hat": 1, "ball": 0}, "A", "B")
    assert result["score"] is None
    assert result["parse_ok"] is True


def test_judge_belief_accuracy_scores_only_the_latest_belief():
    fake = _with_fake_judge_llm("SCORE: 4\nREASONING: Correctly identifies balls as B's top priority.")
    history = ["B might like books.", "B actually seems to prioritize balls highly."]
    result = judge_belief_accuracy(history, {"book": 0, "hat": 1, "ball": 4}, "A", "B")
    assert fake.calls == 1
    assert result["score"] == 4
