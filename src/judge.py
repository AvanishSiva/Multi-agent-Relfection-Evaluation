import re

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

MODEL = "qwen2.5:14b"
# Lower temperature than the negotiation LLM (0.7) — judging should be consistent, not creative.
judge_llm = ChatOllama(model=MODEL, temperature=0.2, max_tokens=200)

_SCORE_RE = re.compile(r"SCORE\s*:\s*", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING\s*:\s*", re.IGNORECASE)


def parse_judge_output(output: str) -> dict:
    # Never raises: malformed judge output falls back to score=None with parse_ok=False.
    score_match = _SCORE_RE.search(output)
    if score_match is None:
        return {"score": None, "reasoning": output.strip(), "parse_ok": False}

    reasoning_match = _REASONING_RE.search(output, score_match.end())
    score_text_end = reasoning_match.start() if reasoning_match else len(output)
    score_text = output[score_match.end():score_text_end]

    number_match = re.search(r"-?\d+", score_text)
    if number_match is None:
        return {"score": None, "reasoning": output.strip(), "parse_ok": False}

    score = int(number_match.group())
    if score < 1 or score > 5:
        return {"score": None, "reasoning": output.strip(), "parse_ok": False}

    reasoning = output[reasoning_match.end():].strip() if reasoning_match else ""
    return {"score": score, "reasoning": reasoning, "parse_ok": True}


def build_belief_consistency_judge_prompt(belief_history: list[str], agent: str) -> str:
    numbered = "\n".join(f"{i + 1}. {belief}" for i, belief in enumerate(belief_history))

    return f"""You are evaluating Agent {agent}'s belief history from a negotiation. Each entry below is a snapshot of what Agent {agent} believed about their opponent's values, in chronological order.

Belief history:
{numbered}

Judge whether Agent {agent}'s beliefs evolved consistently. A RATIONAL UPDATE is a change in belief that reasonably refines or corrects an earlier belief as more evidence came in. A CONTRADICTION is a belief that directly conflicts with an earlier one without acknowledging or justifying the change — including flip-flopping back and forth between claims.

Respond with exactly two lines, in this exact format:
SCORE: <a single integer from 1 to 5, where 5 = fully consistent, only rational updates; 1 = frequent unjustified contradictions>
REASONING: <one or two sentences explaining the score, citing specific belief numbers if relevant>
"""


def build_coherence_judge_prompt(transcript: list[dict], agent: str) -> str:
    lines = []
    for i, entry in enumerate(transcript):
        if entry["action_type"] == "ACCEPT":
            action_str = "ACCEPT"
        else:
            action_str = f"PROPOSE {entry['proposal']}" if entry["proposal"] else "PROPOSE (unparsed)"
        lines.append(f"{i + 1}. Agent {entry['agent']}: \"{entry['message']}\" -> {action_str}")
    numbered = "\n".join(lines)

    return f"""You are evaluating Agent {agent}'s dialogue coherence in the negotiation transcript below (both agents' turns are shown, in order, for context).

Transcript:
{numbered}

For Agent {agent}'s turns only, judge two things: (1) RELEVANCE — does each message actually respond to what the opponent just said or offered, rather than ignoring it? (2) MESSAGE-ACTION MATCH — does the action (the PROPOSE numbers or ACCEPT) actually match what the message claims (e.g. a message saying "I'll compromise" followed by an unchanged offer would be a mismatch)?

Respond with exactly two lines, in this exact format:
SCORE: <a single integer from 1 to 5, where 5 = fully relevant and consistent; 1 = frequently irrelevant or message/action mismatches>
REASONING: <one or two sentences explaining the score, citing specific turn numbers if relevant>
"""


def judge_belief_consistency(belief_history: list[str], agent: str) -> dict:
    if len(belief_history) < 2:
        return {"score": None, "reasoning": "Not enough belief history to judge (fewer than 2 entries).", "parse_ok": True}

    prompt = build_belief_consistency_judge_prompt(belief_history, agent)
    output = judge_llm.invoke(prompt).content
    return parse_judge_output(output)


def judge_dialogue_coherence(transcript: list[dict], agent: str) -> dict:
    if not any(entry["agent"] == agent for entry in transcript):
        return {"score": None, "reasoning": "No turns from this agent to judge.", "parse_ok": True}

    prompt = build_coherence_judge_prompt(transcript, agent)
    output = judge_llm.invoke(prompt).content
    return parse_judge_output(output)
