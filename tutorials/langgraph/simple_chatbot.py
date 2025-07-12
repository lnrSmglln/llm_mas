'''
LangGraph Tutorial - How to Build Advanced AI Agent Systems
https://www.youtube.com/watch?v=1w5cCXlh7JQ&t=65s

Here was Simple LangGraph Chatbot with my additions
'''

from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

load_dotenv()  # take environment variables from .env

llm = init_chat_model(
    model="deepseek-r1:8b",
    model_provider="ollama",
    reasoning=False,
    temperature=0.0
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

'''
 START
   |
   V
chatbot
   |
   V
  END
'''
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, end_key="chatbot")
graph_builder.add_edge(start_key="chatbot", end_key=END)

graph = graph_builder.compile()

user_input = input(">> ")
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(state["messages"][-1].content)