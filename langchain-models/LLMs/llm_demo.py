from langchain_openai import OpenAI
from langchain_groq import ChatGroq
import json
from langchain_google_genai import ChatGoogleGenerativeAI 

from dotenv import load_dotenv
import os

load_dotenv()

# model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROK_API_KEY"), temperature=0)

# # response = model.invoke("What is the capital of France?")
# response = model.invoke("Write 5 lines of poem on Cricket")
# response = model.invoke("Generate a code to find the factorial of a number in Python")

# # print(json.dumps(response.model_dump(), indent=2))
# print(response.content)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

gemini_response = model.invoke("Write 5 lines of poem on Cricket")

print(json.dumps(gemini_response.model_dump(), indent=2))