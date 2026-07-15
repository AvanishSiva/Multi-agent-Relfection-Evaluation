from typing import TypedDict

#State for Negotiation
class NegotiationState(TypedDict):
    pool : dict[str, int] #{"book": 2, "hat": 2, "ball":2}
    a_values: dict[str, int] # Agent A's private valuation
    b_values: dict[str, int] # Agent B's private valuation
    round_count: int #number of rounds that have passed
    max_rounds: int #maximum number of rounds allowed
    condition: str #baseline/reflection/control
    current_proposal: dict[str, int] | None #Agents current proposal, if any
    proposed_by : str | None #Agent who made the current proposal, if any
    a_belief : str # Agent A's belief about the negotiation
    b_belief : str
    a_belief_history : list[str] #Agent A's belief history
    b_belief_history : list[str] #Agent B's belief history
    transcript : list[dict] #  Each entry: {round, agent, message, action}
    accepted : bool # Whether the negotiation has been accepted
    final_allocation : dict[str, int] | None # Final allocation of items if accepted
    a_score : int #Agent A's score
    b_score : int #Agent B's score


# here optimal_joint is maximum possible joint score for the game instance, used for evaluation
GAME_INSTANCES = {
    1: {
        "a_values": {"book": 4, "hat": 1, "ball": 0},
        "b_values": {"book": 0, "hat": 1, "ball": 4},
        "optimal_joint": 18,
        "description": "Integrative: A wants books, B wants balls",
    },
    2: {
        "a_values": {"book": 3, "hat": 2, "ball": 0},
        "b_values": {"book": 0, "hat": 2, "ball": 3},
        "optimal_joint": 16,
        "description": "Mixed: trade exists but both partly want hats",
    },
    3: {
        "a_values": {"book": 4, "hat": 1, "ball": 0},
        "b_values": {"book": 4, "hat": 1, "ball": 0},
        "optimal_joint": 10,
        "description": "Competitive: identical preferences, no win-win possible",
    },
}

POOL = {"book": 2, "hat": 2, "ball": 2}

def get_initial_state(instance_id: int, max_rounds: int, condition: str) -> NegotiationState:
    instance = GAME_INSTANCES[instance_id]
    return NegotiationState(
        pool=POOL.copy(),
        a_values=instance["a_values"].copy(),
        b_values=instance["b_values"].copy(),
        round_count=0,
        max_rounds=max_rounds,
        condition=condition,
        current_proposal=None,
        proposed_by=None,
        a_belief="UNKNOWN",
        b_belief="UNKNOWN",
        a_belief_history=[],
        b_belief_history=[],
        transcript=[],
        accepted=False,
        final_allocation=None,
        a_score=0,
        b_score=0
    )


#Utility Validators
def compute_scores(proposal: dict[str, int], a_values: dict[str, int], b_values: dict[str, int]) -> tuple[int, int]:
    a_score = sum(proposal[item] * a_values[item] for item in proposal)
    b_share = {item: POOL[item] - proposal[item] for item in proposal}
    b_score = sum(b_share[item] * b_values[item] for item in b_share)
    return a_score, b_score


def compute_joint_score(proposal: dict[str, int], a_values: dict[str, int], b_values: dict[str, int]) -> int:
    a_score, b_score = compute_scores(proposal, a_values, b_values)
    return a_score + b_score


def compute_efficiency(a_score: int, b_score: int, optimal_joint: int) -> float:
    return (a_score + b_score) / optimal_joint


def compute_fairness_gap(a_score: int, b_score: int) -> int:
    return abs(a_score - b_score)

def find_optimal_joint(instance_id: int) -> int:
    instance = GAME_INSTANCES[instance_id]
    a_values = instance["a_values"]
    b_values = instance["b_values"]
    best = 0
    for book_a in range(POOL["book"] + 1):
        for hat_a in range(POOL["hat"] + 1):
            for ball_a in range(POOL["ball"] + 1):
                proposal = {"book": book_a, "hat": hat_a, "ball": ball_a}
                score = compute_joint_score(proposal, a_values, b_values)
                if score > best:
                    best = score
    return best


def validate_optimal_joints():
    for instance_id in GAME_INSTANCES:
        computed = find_optimal_joint(instance_id)
        stored = GAME_INSTANCES[instance_id]["optimal_joint"]
        assert computed == stored, f"Instance {instance_id}: computed {computed}, expected {stored}"
    print("All optimal_joint values validated.")
