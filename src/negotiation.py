import re

ITEMS = ["book", "hat", "ball"]

_MESSAGE_RE = re.compile(r"MESSAGE\s*:\s*", re.IGNORECASE)
_ACTION_RE = re.compile(r"ACTION\s*:\s*", re.IGNORECASE)
_BELIEF_RE = re.compile(r"BELIEF\s*:\s*", re.IGNORECASE)
_STRIP_CHARS = " *\n\t"


def _split_message_and_action(output: str) -> tuple[str, str | None]:
    action_match = _ACTION_RE.search(output)
    if action_match is None:
        return output.strip(_STRIP_CHARS), None

    message_match = _MESSAGE_RE.search(output, 0, action_match.start())
    message_start = message_match.end() if message_match else 0
    message = output[message_start:action_match.start()]
    action_text = output[action_match.end():]
    return message.strip(_STRIP_CHARS), action_text.strip(_STRIP_CHARS)


def _extract_proposal(action_text: str, pool: dict[str, int]) -> dict[str, int] | None:
    proposal = {}
    for item in ITEMS:
        match = re.search(rf"{item}_a\s*:\s*(-?\d+)", action_text, re.IGNORECASE)
        if match is None:
            return None
        value = int(match.group(1))
        if value < 0 or value > pool[item]:
            return None
        proposal[item] = value
    return proposal


def _even_split(pool: dict[str, int]) -> dict[str, int]:
    return {item: pool[item] // 2 for item in ITEMS}


def parse_action(
    output: str,
    pool: dict[str, int],
    previous_proposal: dict[str, int] | None,
) -> dict:
    # Never raises: malformed LLM output falls back to the previous proposal (or an even split, on turn 1) with parse_ok=False.
    message, action_text = _split_message_and_action(output)

    if action_text is not None:
        if re.match(r"ACCEPT", action_text, re.IGNORECASE):
            return {
                "message": message,
                "action_type": "ACCEPT",
                "proposal": None,
                "parse_ok": True,
            }

        if re.match(r"PROPOSE", action_text, re.IGNORECASE):
            proposal = _extract_proposal(action_text, pool)
            if proposal is not None:
                return {
                    "message": message,
                    "action_type": "PROPOSE",
                    "proposal": proposal,
                    "parse_ok": True,
                }

    return {
        "message": message,
        "action_type": "PROPOSE",
        "proposal": previous_proposal or _even_split(pool),
        "parse_ok": False,
    }


def parse_belief(output: str, previous_belief: str) -> dict:
    match = _BELIEF_RE.search(output)
    if match is None:
        return {"belief": previous_belief, "parse_ok": False}

    belief = output[match.end():].strip(_STRIP_CHARS)
    if not belief:
        return {"belief": previous_belief, "parse_ok": False}

    return {"belief": belief, "parse_ok": True}
