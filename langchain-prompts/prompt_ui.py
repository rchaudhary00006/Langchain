import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import json
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
client = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROK_API_KEY"), temperature=0,
                  reasoning_effort="medium"
                  )

st.header("LangChain Groq Integration")

paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "CodeOriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

prompt_template = load_prompt("prompt_template.json")

prompt = prompt_template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Submit"):
    response = client.invoke(prompt)
    st.write(f"Response: {response.content}")
    # st.write(f"Response: {response}")

