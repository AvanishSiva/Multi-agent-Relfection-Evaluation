from langchain_ollama import ChatOllama

from src.negotiation import parse_action
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
