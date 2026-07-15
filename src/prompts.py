from src.negotiation import ITEMS


def _format_values(values: dict[str, int]) -> str:
    return ", ".join(f"{item}: {values[item]} points each" for item in ITEMS)


def _format_pool(pool: dict[str, int]) -> str:
    return ", ".join(f"{pool[item]} {item}(s)" for item in ITEMS)


def _format_standing_proposal(state: dict) -> str:
    proposal = state["current_proposal"]
    if proposal is None:
        return "No proposal has been made yet."

    pool = state["pool"]
    b_share = {item: pool[item] - proposal[item] for item in ITEMS}
    proposer = state["proposed_by"]
    a_share_str = ", ".join(f"{item}: {proposal[item]}" for item in ITEMS)
    b_share_str = ", ".join(f"{item}: {b_share[item]}" for item in ITEMS)
    return (
        f"Proposed by Agent {proposer}. If accepted: Agent A gets ({a_share_str}), "
        f"Agent B gets ({b_share_str})."
    )


def _format_transcript(transcript: list[dict]) -> str:
    if not transcript:
        return "(No turns yet — you may be the first to speak.)"

    lines = []
    for entry in transcript:
        if entry["action_type"] == "ACCEPT":
            action_str = "ACCEPT"
        else:
            proposal = entry["proposal"]
            action_str = "PROPOSE " + ", ".join(f"{item}_A: {proposal[item]}" for item in ITEMS)
        lines.append(f"Round {entry['round']} - Agent {entry['agent']}: {entry['message']} [{action_str}]")
    return "\n".join(lines)


def build_agent_prompt(state: dict, agent: str) -> str:
    opponent = "B" if agent == "A" else "A"
    own_values = state["a_values"] if agent == "A" else state["b_values"]
    pool = state["pool"]

    return f"""You are Agent {agent} negotiating with Agent {opponent} over how to split a pool of items. All items must be allocated — nothing is left over, nothing is split in half.

Pool: {_format_pool(pool)}

Your private values (Agent {opponent} does NOT know these): {_format_values(own_values)}
You do NOT know Agent {opponent}'s values.

Round {state['round_count'] + 1} of {state['max_rounds']}.

Standing proposal: {_format_standing_proposal(state)}

Conversation so far:
{_format_transcript(state['transcript'])}

Respond with exactly two lines, in this exact format:
MESSAGE: <one or two sentences of reasoning, an offer, or a response>
ACTION: PROPOSE {{book_A: N, hat_A: N, ball_A: N}}
(or, to close the deal on the standing proposal instead:)
ACTION: ACCEPT

Rules:
- PROPOSE always states how many of each item AGENT A receives, even if you are Agent B — the numbers always describe Agent A's share; the other agent receives everything else.
- Each value must be a whole number: 0-{pool['book']} for book, 0-{pool['hat']} for hat, 0-{pool['ball']} for ball.
- Use ACTION: ACCEPT only if there is a standing proposal you want to accept.
"""
