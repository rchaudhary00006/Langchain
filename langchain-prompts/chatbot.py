from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROK_API_KEY"), temperature=0,
                  reasoning_effort="medium"
                  )

messages = [SystemMessage(content="You are a helpful assistant")]

while True:
    user_query = input("You: ")

    if(user_query == "exit"):
        break
    messages.append(HumanMessage(content=user_query))

    response = model.invoke(messages)
    print("AI: ", response.content)
    messages.append(AIMessage(content=response.content))

print(messages)
