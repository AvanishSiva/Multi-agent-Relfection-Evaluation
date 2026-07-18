from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from src.negotiation import parse_action, parse_belief
from src.negotiation_state import NegotiationState, compute_scores
from src.prompts import build_agent_prompt, build_control_prompt, build_reflection_prompt

MODEL = "qwen2.5:14b"
llm = ChatOllama(model=MODEL, temperature=0.7, max_tokens=300)


def _propose(state: dict, agent: str) -> dict:
    prompt = build_agent_prompt(state, agent)
    output = llm.invoke(prompt).content
    result = parse_action(output, state["pool"], state["current_proposal"])

    entry = {
        "round": state["round_count"] + 1,
        "agent": agent,
        "message": result["message"],
        "action_type": result["action_type"],
        "proposal": result["proposal"],
        "parse_ok": result["parse_ok"],
    }
    update = {"transcript": state["transcript"] + [entry]}

    if result["action_type"] == "ACCEPT":
        update["accepted"] = True
    else:
        update["current_proposal"] = result["proposal"]
        update["proposed_by"] = agent

    return update


def a_propose(state: dict) -> dict:
    return _propose(state, "A")


def b_propose(state: dict) -> dict:
    update = _propose(state, "B")
    update["round_count"] = state["round_count"] + 1
    return update


def _reflect(state: dict, agent: str) -> dict:
    if state["condition"] == "reflection":
        prompt = build_reflection_prompt(state, agent)
    else:
        prompt = build_control_prompt(state, agent)

    output = llm.invoke(prompt).content
    belief_key = "a_belief" if agent == "A" else "b_belief"
    history_key = "a_belief_history" if agent == "A" else "b_belief_history"
    result = parse_belief(output, state[belief_key])

    return {
        belief_key: result["belief"],
        history_key: state[history_key] + [result["belief"]],
    }


def a_reflect(state: dict) -> dict:
    return _reflect(state, "A")


def b_reflect(state: dict) -> dict:
    return _reflect(state, "B")


def round_start(state: dict) -> dict:
    return {}


def evaluate(state: dict) -> dict:
    if state["accepted"]:
        final_allocation = state["current_proposal"]
        a_score, b_score = compute_scores(final_allocation, state["a_values"], state["b_values"])
        return {
            "final_allocation": final_allocation,
            "a_score": a_score,
            "b_score": b_score,
        }

    if state["round_count"] >= state["max_rounds"]:
        return {
            "final_allocation": None,
            "a_score": 0,
            "b_score": 0,
        }

    return {}


def keep_going(state: dict) -> str:
    if state["accepted"] or state["round_count"] >= state["max_rounds"]:
        return "end"
    return "continue"


def route_before_a(state: dict) -> str:
    return "a_propose" if state["condition"] == "baseline" else "a_reflect"


def route_after_a(state: dict) -> str:
    # If A just ACCEPTed, the deal closes immediately — B does not get a turn this round.
    if state["accepted"]:
        return "evaluate"
    return "b_propose" if state["condition"] == "baseline" else "b_reflect"


def build_graph():
    g = StateGraph(NegotiationState)
    g.add_node("round_start", round_start)
    g.add_node("a_reflect", a_reflect)
    g.add_node("a_propose", a_propose)
    g.add_node("b_reflect", b_reflect)
    g.add_node("b_propose", b_propose)
    g.add_node("evaluate", evaluate)

    g.add_edge(START, "round_start")
    g.add_conditional_edges(
        "round_start",
        route_before_a,
        {"a_propose": "a_propose", "a_reflect": "a_reflect"},
    )
    g.add_edge("a_reflect", "a_propose")
    g.add_conditional_edges(
        "a_propose",
        route_after_a,
        {"b_propose": "b_propose", "b_reflect": "b_reflect", "evaluate": "evaluate"},
    )
    g.add_edge("b_reflect", "b_propose")
    g.add_edge("b_propose", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        keep_going,
        {"continue": "round_start", "end": END},
    )

    return g.compile()
