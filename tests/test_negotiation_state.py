import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.negotiation_state import (
    compute_joint_score,
    compute_scores,
    compute_efficiency,
    compute_fairness_gap,
    find_optimal_joint,
    get_initial_state,
    validate_optimal_joints,
    GAME_INSTANCES,
    POOL,
)



def test_compute_joint_score_instance1_optimal():
    # A gets both books: optimal split for instance 1
    proposal = {"book": 2, "hat": 0, "ball": 0}
    a_values = GAME_INSTANCES[1]["a_values"]
    b_values = GAME_INSTANCES[1]["b_values"]
    assert compute_joint_score(proposal, a_values, b_values) == 18

def test_compute_joint_score_no_deal():
    # A gets nothing, B gets everything
    proposal = {"book": 0, "hat": 0, "ball": 0}
    a_values = GAME_INSTANCES[1]["a_values"]
    b_values = GAME_INSTANCES[1]["b_values"]
    # b gets all: (0*2) + (1*2) + (4*2) = 10; a gets 0
    assert compute_joint_score(proposal, a_values, b_values) == 10

def test_compute_joint_score_instance3_best():
    # Instance 3: identical preferences — best one agent can get is 9 (book:2, hat:1)
    proposal = {"book": 2, "hat": 1, "ball": 0}
    a_values = GAME_INSTANCES[3]["a_values"]
    b_values = GAME_INSTANCES[3]["b_values"]
    # a: (4*2)+(1*1)+(0*0)=9; b: (4*0)+(1*1)+(0*2)=1 → joint=10
    assert compute_joint_score(proposal, a_values, b_values) == 10



def test_find_optimal_joint_instance1():
    assert find_optimal_joint(1) == 18

def test_find_optimal_joint_instance2():
    assert find_optimal_joint(2) == 16

def test_find_optimal_joint_instance3():
    # Sanity check: competitive game cannot exceed 10
    assert find_optimal_joint(3) == 10

def test_find_optimal_joint_matches_stored():
    for instance_id in GAME_INSTANCES:
        computed = find_optimal_joint(instance_id)
        stored = GAME_INSTANCES[instance_id]["optimal_joint"]
        assert computed == stored, f"Instance {instance_id}: computed {computed}, stored {stored}"



def test_get_initial_state_defaults():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert state["round_count"] == 0
    assert state["accepted"] == False
    assert state["current_proposal"] is None
    assert state["proposed_by"] is None
    assert state["final_allocation"] is None
    assert state["a_score"] == 0
    assert state["a_belief_parse_ok"] == []
    assert state["b_belief_parse_ok"] == []
    assert state["b_score"] == 0

def test_get_initial_state_histories_empty():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert state["a_belief_history"] == []
    assert state["b_belief_history"] == []
    assert state["transcript"] == []

def test_get_initial_state_belief_unknown():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert state["a_belief"] == "UNKNOWN"
    assert state["b_belief"] == "UNKNOWN"

def test_get_initial_state_pool():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert state["pool"] == {"book": 2, "hat": 2, "ball": 2}

def test_get_initial_state_values_loaded():
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    assert state["a_values"] == {"book": 4, "hat": 1, "ball": 0}
    assert state["b_values"] == {"book": 0, "hat": 1, "ball": 4}

def test_get_initial_state_condition():
    for condition in ["baseline", "reflection", "control"]:
        state = get_initial_state(instance_id=1, max_rounds=8, condition=condition)
        assert state["condition"] == condition

def test_get_initial_state_pool_is_copy():
    # Mutating one state's pool should not affect POOL constant
    state = get_initial_state(instance_id=1, max_rounds=8, condition="baseline")
    state["pool"]["book"] = 99
    assert POOL["book"] == 2


# --- validate_optimal_joints ---

def test_validate_optimal_joints_passes():
    validate_optimal_joints()


# --- compute_scores / compute_efficiency / compute_fairness_gap ---

def test_compute_scores_instance1_optimal_split():
    proposal = {"book": 2, "hat": 0, "ball": 0}
    a_values = GAME_INSTANCES[1]["a_values"]
    b_values = GAME_INSTANCES[1]["b_values"]
    a_score, b_score = compute_scores(proposal, a_values, b_values)
    assert a_score == 8
    assert b_score == 10
    assert a_score + b_score == compute_joint_score(proposal, a_values, b_values)

def test_compute_efficiency_optimal_split_is_1():
    a_score, b_score = 8, 10
    optimal_joint = GAME_INSTANCES[1]["optimal_joint"]
    assert compute_efficiency(a_score, b_score, optimal_joint) == 1.0

def test_compute_efficiency_no_deal_is_0():
    assert compute_efficiency(0, 0, GAME_INSTANCES[1]["optimal_joint"]) == 0.0

def test_compute_fairness_gap():
    assert compute_fairness_gap(8, 10) == 2
    assert compute_fairness_gap(5, 5) == 0
