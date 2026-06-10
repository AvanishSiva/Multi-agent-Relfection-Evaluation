import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

MODEL = "qwen2.5:14b"
llm = ChatOllama(model=MODEL, temperature=0.7, max_tokens=1000)



class ChatState(TypedDict):
    log: Annotated[list, operator.add]
    turn: int
    max_turns: int

def history(state):
    return "\n".join(state["log"]) if state["log"] else "(start of conversation)"

def alice(state: ChatState):
    prompt = (f"You are Alice. Have a friendly chat. Reply in ONE short sentence.\n\n"
              f"Conversation so far:\n{history(state)}\n\nAlice:")
    response = llm.invoke(prompt).content.strip()
    print(f"Alice: {response}")
    return {"log" : [f"Alice: {response}"]}

def bob(state: ChatState):
    prompt = (f"You are Bob. Have a friendly chat. Reply in ONE short sentence.\n\n"
              f"Conversation so far:\n{history(state)}\n\nBob:")
    reply = llm.invoke(prompt).content.strip()
    print(f"  Bob:   {reply}")
    return {"log": [f"Bob: {reply}"], "turn": state["turn"] + 1}
 
def keep_going(state: ChatState):
    return "stop" if state["turn"] > state["max_turns"] else "continue"
 

def build():
    g = StateGraph(ChatState)

    g.add_node("alice", alice)
    g.add_node("bob", bob)

    g.add_edge(START, "alice")
    g.add_edge("alice", "bob")

    g.add_conditional_edges(
        "bob",
        keep_going,
        {
            "continue": "alice",
            "stop": END,
        },
    )

    return g.compile()

def main():
    print(f"Two agents chatting with {MODEL} (first reply is slow)...\n")
    build().invoke({"log": [], "turn": 1, "max_turns": 10},
                   config={"recursion_limit": 50})
    print("\nMULTI-AGENT OK — two agents took turns and saw each other's messages.")
 

if __name__ == "__main__":
    main()