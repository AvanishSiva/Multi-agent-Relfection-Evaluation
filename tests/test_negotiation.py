import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.negotiation import parse_action, parse_belief

POOL = {"book": 2, "hat": 2, "ball": 2}


def test_parse_well_formed_propose():
    output = "MESSAGE: I really only care about the books.\nACTION: PROPOSE {book_A: 2, hat_A: 0, ball_A: 0}"
    result = parse_action(output, POOL, previous_proposal=None)
    assert result["parse_ok"] is True
    assert result["action_type"] == "PROPOSE"
    assert result["proposal"] == {"book": 2, "hat": 0, "ball": 0}
    assert "books" in result["message"]


def test_parse_well_formed_accept():
    output = "MESSAGE: That works for me.\nACTION: ACCEPT"
    result = parse_action(output, POOL, previous_proposal={"book": 1, "hat": 1, "ball": 1})
    assert result["parse_ok"] is True
    assert result["action_type"] == "ACCEPT"
    assert result["proposal"] is None


def test_parse_missing_action_line_falls_back():
    output = "MESSAGE: I am thinking about it."
    previous = {"book": 1, "hat": 1, "ball": 0}
    result = parse_action(output, POOL, previous_proposal=previous)
    assert result["parse_ok"] is False
    assert result["action_type"] == "PROPOSE"
    assert result["proposal"] == previous


def test_parse_proposal_exceeds_pool_falls_back():
    output = "MESSAGE: Give me everything.\nACTION: PROPOSE {book_A: 5, hat_A: 0, ball_A: 0}"
    previous = {"book": 0, "hat": 0, "ball": 0}
    result = parse_action(output, POOL, previous_proposal=previous)
    assert result["parse_ok"] is False
    assert result["proposal"] == previous


def test_parse_proposal_missing_item_falls_back():
    output = "MESSAGE: Here is my offer.\nACTION: PROPOSE {book_A: 1, hat_A: 1}"
    previous = {"book": 2, "hat": 0, "ball": 0}
    result = parse_action(output, POOL, previous_proposal=previous)
    assert result["parse_ok"] is False
    assert result["proposal"] == previous


def test_parse_non_integer_value_falls_back():
    output = "MESSAGE: Here is my offer.\nACTION: PROPOSE {book_A: two, hat_A: 0, ball_A: 0}"
    previous = {"book": 1, "hat": 1, "ball": 1}
    result = parse_action(output, POOL, previous_proposal=previous)
    assert result["parse_ok"] is False
    assert result["proposal"] == previous


def test_parse_negative_value_falls_back():
    output = "MESSAGE: Here is my offer.\nACTION: PROPOSE {book_A: -1, hat_A: 1, ball_A: 1}"
    previous = {"book": 1, "hat": 1, "ball": 1}
    result = parse_action(output, POOL, previous_proposal=previous)
    assert result["parse_ok"] is False
    assert result["proposal"] == previous


def test_parse_extra_whitespace_and_lowercase():
    output = "message:   I'll take the hats.\naction:   propose {book_A: 0, hat_A: 2, ball_A: 0}"
    result = parse_action(output, POOL, previous_proposal=None)
    assert result["parse_ok"] is True
    assert result["proposal"] == {"book": 0, "hat": 2, "ball": 0}


def test_parse_markdown_bold_labels():
    output = "**MESSAGE:** Splitting evenly seems fair.\n**ACTION:** PROPOSE {book_A: 1, hat_A: 1, ball_A: 1}"
    result = parse_action(output, POOL, previous_proposal=None)
    assert result["parse_ok"] is True
    assert result["proposal"] == {"book": 1, "hat": 1, "ball": 1}
    assert "Splitting evenly" in result["message"]


def test_parse_no_previous_proposal_falls_back_to_even_split():
    output = "MESSAGE: Not sure yet."
    result = parse_action(output, POOL, previous_proposal=None)
    assert result["parse_ok"] is False
    assert result["proposal"] == {"book": 1, "hat": 1, "ball": 1}


def test_parse_unrecognized_action_falls_back():
    output = "MESSAGE: Let's keep talking.\nACTION: THINK_ABOUT_IT"
    previous = {"book": 1, "hat": 0, "ball": 1}
    result = parse_action(output, POOL, previous_proposal=previous)
    assert result["parse_ok"] is False
    assert result["proposal"] == previous


# --- parse_belief ---

def test_parse_belief_well_formed():
    output = "BELIEF: Agent B seems to value balls far more than hats."
    result = parse_belief(output, previous_belief="UNKNOWN")
    assert result["parse_ok"] is True
    assert result["belief"] == "Agent B seems to value balls far more than hats."


def test_parse_belief_missing_label_falls_back():
    output = "I think B likes balls."
    result = parse_belief(output, previous_belief="UNKNOWN")
    assert result["parse_ok"] is False
    assert result["belief"] == "UNKNOWN"


def test_parse_belief_case_insensitive_and_whitespace_stripped():
    output = "belief:   B still seems focused on balls.   \n"
    result = parse_belief(output, previous_belief="UNKNOWN")
    assert result["parse_ok"] is True
    assert result["belief"] == "B still seems focused on balls."


def test_parse_belief_empty_after_label_falls_back():
    output = "BELIEF:    "
    previous = "B wants balls."
    result = parse_belief(output, previous_belief=previous)
    assert result["parse_ok"] is False
    assert result["belief"] == previous
