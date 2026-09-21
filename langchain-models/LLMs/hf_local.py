from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import json

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_HOME'] = '/home/rahulchaudhari/AI/Learning Python/huggingface_cache'

llm = HuggingFacePipeline.from_model_id (
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=100,
        temperature=0.7,
    )
)

model = ChatHuggingFace(llm=llm)

model_response = model.invoke("What is the capital of India?");

print(model_response.content)

