import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.negotiation_state import get_initial_state
from src.prompts import build_agent_prompt, build_reflection_prompt, build_control_prompt


def test_reflection_prompt_includes_own_values_and_belief_not_opponent_values():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    state["a_belief"] = "B seems to want balls more than hats."

    prompt = build_reflection_prompt(state, "A")

    assert "book: 4 points each" in prompt  # A's own value
    assert "B seems to want balls more than hats." in prompt
    assert "book: 0 points each" not in prompt  # B's own value, must not leak


def test_reflection_prompt_handles_unknown_belief_on_first_round():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    prompt = build_reflection_prompt(state, "A")
    assert "UNKNOWN" in prompt


def test_reflection_prompt_ends_with_belief_output_format():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    prompt = build_reflection_prompt(state, "B")
    assert "BELIEF:" in prompt


def test_control_prompt_omits_values_and_belief_section():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="control")
    prompt = build_control_prompt(state, "A")
    assert "Your private values" not in prompt
    assert "Your current belief" not in prompt


def test_control_prompt_ends_with_same_belief_output_format_as_reflection():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="control")
    control_prompt = build_control_prompt(state, "A")
    reflection_prompt = build_reflection_prompt(state, "A")
    assert "BELIEF:" in control_prompt
    assert "BELIEF:" in reflection_prompt


# --- build_agent_prompt belief-line gating ---

def test_agent_prompt_baseline_omits_belief_line():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    prompt = build_agent_prompt(state, "A")
    assert "current belief" not in prompt


def test_agent_prompt_reflection_includes_belief_line():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="reflection")
    state["a_belief"] = "B seems to want balls."
    prompt = build_agent_prompt(state, "A")
    assert "Your current belief about Agent B's values (from your reflection): B seems to want balls." in prompt


def test_agent_prompt_control_includes_belief_line():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="control")
    state["b_belief"] = "A seems to want books."
    prompt = build_agent_prompt(state, "B")
    assert "Your current belief about Agent A's values (from your reflection): A seems to want books." in prompt


def test_agent_prompt_baseline_is_unchanged_by_condition_branch():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    prompt = build_agent_prompt(state, "A")
    # No blank-line artifacts from an empty belief_line insertion.
    assert "\n\n\n" not in prompt
