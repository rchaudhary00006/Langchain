from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

model_response = model.invoke("Write 5 lines of poem on Cricket")

print(model_response.content)
# print(json.dumps(model_response.model_dump(), indent=2))