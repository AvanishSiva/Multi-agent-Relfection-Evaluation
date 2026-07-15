from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from src.negotiation import parse_action
from src.negotiation_state import NegotiationState, compute_scores
from src.prompts import build_agent_prompt

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


def route_after_a(state: dict) -> str:
    # If A just ACCEPTed, the deal closes immediately — B does not get a turn this round.
    return "evaluate" if state["accepted"] else "b_propose"


def build_baseline_graph():
    g = StateGraph(NegotiationState)
    g.add_node("a_propose", a_propose)
    g.add_node("b_propose", b_propose)
    g.add_node("evaluate", evaluate)

    g.add_edge(START, "a_propose")
    g.add_conditional_edges(
        "a_propose",
        route_after_a,
        {"b_propose": "b_propose", "evaluate": "evaluate"},
    )
    g.add_edge("b_propose", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        keep_going,
        {"continue": "a_propose", "end": END},
    )

    return g.compile()
