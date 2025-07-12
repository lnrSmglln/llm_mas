from langchain_core.messages import ChatMessage, HumanMessage
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(
    model="deepseek-r1:8b",
    reasoning=False,
    temperature=0.0
)

messages = [
    # ChatMessage(role="control", content="thinking"),
    HumanMessage(content="What is pydantic?"),
]

response = llm.invoke(messages)

print(response.content)