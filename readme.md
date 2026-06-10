# Scenario Specification — Multi-Agent Negotiation

*The game two AI agents play, written so anyone can follow it.*

---

## 1. The idea in one paragraph

Two AI agents share a small pile of items. They must split **every** item between them by talking it out. Each agent secretly values the items differently, and neither can see the other's values: they have to figure that out from the conversation. If they're smart, both can walk away happy, because they often want different things. If they argue too long and never agree, **both get nothing**. We use this game to test whether giving an agent a moment to *reflect* before it speaks makes it (a) share better and (b) stop contradicting itself.

This is the classic **"Deal or No Deal" multi-issue bargaining task** (Lewis et al., 2017), the standard testbed for this kind of negotiation. We keep the scenario standard on purpose, so that any improvement we measure can be credited to *reflection*, not to a custom game.

---

## 2. The pile (the "pool")

There are **3 item types, 2 of each** — 6 items total:

| Item | Count |
|------|-------|
| Books | 2 |
| Hats  | 2 |
| Balls | 2 |

**Rule:** every item must go to Agent A or Agent B. Nothing is left over, nothing is split in half.

---

## 3. What each agent wants (goals)

Each agent has a **private valuation**:  how many points each item type is worth *to them*. The other agent never sees these numbers; it can only guess from what's said.

Two fairness rules on the setup:

- An agent's values over the **whole pile always add up to 10**, so both agents care about the pile equally in total.
- Values are fixed for the whole negotiation (an agent doesn't suddenly start liking something new).

**Each agent's goal:** end up with the set of items worth the most points *to itself* — while remembering that walking away with no deal is worth **zero**. So an agent has to balance pushing for what it wants against the risk of getting nothing.

---

## 4. The three game versions ("instances")

We run three fixed versions. Numbers read as `(book, hat, ball)`.

| Version | Agent A values | Agent B values | What's going on | Best possible *joint* score |
|---------|----------------|----------------|-----------------|------------------------------|
| **1 — Integrative** | (4, 1, 0) | (0, 1, 4) | A wants books, B wants balls — they want different things | **18** |
| **2 — Mixed** | (3, 2, 0) | (0, 2, 3) | A trade exists, but both somewhat want hats | **16** |
| **3 — Competitive (control)** | (4, 1, 0) | (4, 1, 0) | They want the **exact same** things | **10** |

**Why version 3 matters (the safety check):** here both agents want identical things, so no clever win-win is possible — the best they can *ever* do together is 10. This is a test of our own measuring tools. If our scoring ever reports "great cooperation" in this version, we know the scoring is broken. It stops us fooling ourselves.

---

## 5. How a turn works (the protocol)

Agents take turns. On each turn, an agent must output **two parts**:

```
MESSAGE: <free text — what it says out loud, e.g. reasoning, offers, questions>
ACTION:  PROPOSE {book_A: 2, hat_A: 1, ball_A: 0}   OR   ACCEPT
```

- **MESSAGE** is the natural-language part — this is where the agent reveals, hides, and reasons (and where we later measure consistency and coherence).
- **ACTION** is the move:
  - `PROPOSE {…}` states exactly how many of each item Agent A would get (Agent B automatically gets the rest).
  - `ACCEPT` means "I agree to your last proposal" — and the deal closes immediately.

The `{…}` always describes A's share; B's share is whatever's left. Example: `PROPOSE {book_A: 2, hat_A: 1, ball_A: 0}` means A takes both books and one hat; B gets both balls and one hat.

**Parsing note:** the `ACTION` line is read by the program. If an agent's output can't be parsed, the system falls back to that agent's previous proposal and logs the miss (parse-success is tracked as a small quality metric).

---

## 6. When the game ends (termination rules)

The negotiation stops in one of two ways:

1. **Deal reached** — one agent plays `ACCEPT` on the other's standing proposal. That split is final, and each agent scores the value of the items it received.
2. **Time runs out** — they reach the **round cap (6–8 rounds)** with no acceptance. This counts as **no deal**, and **both agents score 0**.

There's no partial credit for "almost agreeing." It's a full deal or nothing — that's what creates the pressure to actually cooperate.

---

## 7. What we track during a game (the state)

The shared "memory" passed along as the game runs:

| Field | Meaning |
|-------|---------|
| `pool` | The items available: `{book: 2, hat: 2, ball: 2}` |
| `a_values`, `b_values` | Each agent's private values (never shown to the other) |
| `round`, `max_rounds` | Current turn number and the cap |
| `condition` | Which experiment group: `baseline` / `reflection` / `control` |
| `current_proposal`, `proposed_by` | The latest offer on the table, and who made it |
| `a_belief`, `b_belief` | Each agent's stated guess about what the *other* wants |
| `a_belief_history`, `b_belief_history` | Those guesses over time — used to measure consistency |
| `transcript` | The full running record of messages and actions |
| `accepted`, `final_alloc` | Whether a deal happened, and the final split |
| `a_score`, `b_score` | Final points for each agent |

The two `*_belief_history` fields are the important new part — they're how we check whether an agent kept its story straight.

---

## 8. How we score cooperation

Before the game, the program checks **all 27 possible ways** to split the items (3 choices each for books, hats, balls) and finds the **best possible combined score** — the "optimum" in the table above.

After the game:

```
efficiency = (a_score + b_score) / best_possible_joint_score
```

This is a number from **0 to 1**. If the best possible was 18 and the agents together got 9, efficiency = 0.5 → "they captured half of what was achievable." It lets us compare any two games on one clean scale.

We also track: did they reach a deal at all, how many rounds it took, and the fairness gap `|a_score − b_score|`.

---

## 9. The three conditions (where the experiment happens)

The scenario above is run three ways, changing **only** how an agent thinks before it speaks:

| Condition | What the agent does before each move |
|-----------|--------------------------------------|
| **Baseline** | Nothing extra — it just responds. |
| **Reflection** | Pauses to think first: *Is my position consistent? What does the other agent actually want? Is there a deal that's better for both of us?* This is the thing we're testing. |
| **Control (compute-matched)** | Gets an extra step that is **not** real reflection (e.g. "restate the current situation"): same extra effort, no real thinking. |

The control matters: if reflection beats **both** the baseline *and* this control, we know the gain came from *thinking*, not just from "taking one more step."

---

---

## 10. Why this scenario (one line for the write-up)

It's a standard, well-cited negotiation game with **private values** (so "belief" means something real) and a **computable best outcome** (so "cooperation" is a clean number) which lets us isolate the effect of reflection rather than the effect of a custom environment.

**Reference:** Lewis, M., Yarats, D., Dauphin, Y., Parikh, D., & Batra, D. (2017). *Deal or No Deal? End-to-End Learning for Negotiation Dialogues.* EMNLP 2017.