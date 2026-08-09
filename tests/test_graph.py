import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.graph as graph
from src.negotiation_state import get_initial_state


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


def test_a_propose_appends_transcript_and_sets_proposal():
    _with_fake_llm("MESSAGE: opening offer\nACTION: PROPOSE {book_A: 2, hat_A: 0, ball_A: 0}")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    update = graph.a_propose(state)
    assert len(update["transcript"]) == 1
    assert update["transcript"][0]["agent"] == "A"
    assert update["transcript"][0]["round"] == 1
    assert update["current_proposal"] == {"book": 2, "hat": 0, "ball": 0}
    assert update["proposed_by"] == "A"
    assert "accepted" not in update


def test_a_propose_does_not_mutate_original_transcript():
    _with_fake_llm("MESSAGE: hi\nACTION: PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    original_transcript = state["transcript"]
    graph.a_propose(state)
    assert state["transcript"] is original_transcript
    assert len(original_transcript) == 0


def test_b_propose_does_not_touch_round_count():
    _with_fake_llm("MESSAGE: counter\nACTION: PROPOSE {book_A: 1, hat_A: 0, ball_A: 1}")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["current_proposal"] = {"book": 2, "hat": 0, "ball": 0}
    state["proposed_by"] = "A"
    state["transcript"] = [{
        "round": 1, "agent": "A", "message": "opening",
        "action_type": "PROPOSE", "proposal": {"book": 2, "hat": 0, "ball": 0},
        "parse_ok": True,
    }]

    update = graph.b_propose(state)
    assert "round_count" not in update
    assert len(update["transcript"]) == 2
    assert update["transcript"][1]["agent"] == "B"
    assert update["proposed_by"] == "B"
    assert update["current_proposal"] == {"book": 1, "hat": 0, "ball": 1}


def test_propose_accept_sets_accepted_flag_and_leaves_proposal_unchanged():
    _with_fake_llm("MESSAGE: deal\nACTION: ACCEPT")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["current_proposal"] = {"book": 1, "hat": 1, "ball": 1}
    state["proposed_by"] = "A"

    update = graph.b_propose(state)
    assert update["accepted"] is True
    assert "current_proposal" not in update
    assert update["transcript"][-1]["action_type"] == "ACCEPT"


def test_propose_falls_back_on_malformed_output_and_logs_parse_miss():
    _with_fake_llm("This is not formatted correctly at all")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["current_proposal"] = {"book": 1, "hat": 1, "ball": 1}
    state["proposed_by"] = "B"

    update = graph.a_propose(state)
    assert update["current_proposal"] == {"book": 1, "hat": 1, "ball": 1}
    assert update["transcript"][-1]["parse_ok"] is False


# --- evaluate / keep_going ---

def test_evaluate_on_accept_computes_scores_and_allocation():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["current_proposal"] = {"book": 2, "hat": 0, "ball": 0}
    state["proposed_by"] = "A"
    state["accepted"] = True

    update = graph.evaluate(state)
    assert update["round_count"] == 1
    assert update["final_allocation"] == {"book": 2, "hat": 0, "ball": 0}
    assert update["a_score"] == 8
    assert update["b_score"] == 10


def test_evaluate_increments_round_count_even_when_b_was_skipped():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["round_count"] = 3
    state["current_proposal"] = {"book": 1, "hat": 1, "ball": 1}
    state["accepted"] = True

    update = graph.evaluate(state)
    assert update["round_count"] == 4


def test_evaluate_round_cap_no_deal_scores_zero():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["round_count"] = 7
    state["accepted"] = False

    update = graph.evaluate(state)
    assert update["round_count"] == 8
    assert update["final_allocation"] is None
    assert update["a_score"] == 0
    assert update["b_score"] == 0


def test_evaluate_mid_game_only_increments_round_count():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["round_count"] = 2
    state["accepted"] = False

    update = graph.evaluate(state)
    assert update == {"round_count": 3}


def test_keep_going_continues_mid_game():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["round_count"] = 2
    assert graph.keep_going(state) == "continue"


def test_keep_going_ends_on_accept():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["round_count"] = 2
    state["accepted"] = True
    assert graph.keep_going(state) == "end"


def test_keep_going_ends_on_round_cap():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["round_count"] = 8
    assert graph.keep_going(state) == "end"


def test_route_after_a_goes_to_b_propose_normally():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert graph.route_after_a(state) == "b_propose"


def test_route_after_a_skips_b_when_a_accepts():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["accepted"] = True
    assert graph.route_after_a(state) == "evaluate"


# --- route_before_a / route_after_a condition toggle ---

def test_route_before_a_baseline_skips_reflect():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert graph.route_before_a(state) == "a_propose"


def test_route_before_a_reflection_uses_reflect():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    assert graph.route_before_a(state) == "a_reflect"


def test_route_before_a_control_uses_reflect():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="control")
    assert graph.route_before_a(state) == "a_reflect"


def test_route_after_a_reflection_goes_to_b_reflect():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    assert graph.route_after_a(state) == "b_reflect"


def test_route_after_a_still_skips_b_when_a_accepts_in_reflection_condition():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    state["accepted"] = True
    assert graph.route_after_a(state) == "evaluate"


# --- a_reflect / b_reflect ---

def test_a_reflect_updates_belief_and_appends_history():
    _with_fake_llm("BELIEF: Agent B seems to want balls, not hats.")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")

    update = graph.a_reflect(state)
    assert update["a_belief"] == "Agent B seems to want balls, not hats."
    assert update["a_belief_history"] == ["Agent B seems to want balls, not hats."]
    assert update["a_belief_parse_ok"] == [True]


def test_a_reflect_does_not_mutate_original_history():
    _with_fake_llm("BELIEF: some updated belief")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    original_history = state["a_belief_history"]

    graph.a_reflect(state)
    assert state["a_belief_history"] is original_history
    assert len(original_history) == 0


def test_b_reflect_updates_belief_and_appends_history():
    _with_fake_llm("BELIEF: Agent A seems to want books.")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="control")

    update = graph.b_reflect(state)
    assert update["b_belief"] == "Agent A seems to want books."
    assert update["b_belief_history"] == ["Agent A seems to want books."]
    assert update["b_belief_parse_ok"] == [True]


def test_reflect_falls_back_to_previous_belief_on_malformed_output():
    _with_fake_llm("This response forgot the required format.")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    state["a_belief"] = "B wants balls."

    update = graph.a_reflect(state)
    assert update["a_belief"] == "B wants balls."
    assert update["a_belief_history"] == ["B wants balls."]
    assert update["a_belief_parse_ok"] == [False]


def test_a_reflect_appends_to_existing_parse_ok_history():
    _with_fake_llm("BELIEF: updated belief")
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    state["a_belief_parse_ok"] = [True, False]

    update = graph.a_reflect(state)
    assert update["a_belief_parse_ok"] == [True, False, True]
