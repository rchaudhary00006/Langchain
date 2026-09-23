from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnablePassthrough
import os
from langchain_groq import ChatGroq

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROK_API_KEY"), temperature=0)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate summary on the topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text {topic}",
    input_variables=['topic']
)

summary_generator_chain = RunnableSequence(prompt1, model, parser)

summarize_text_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(summary_generator_chain, summarize_text_chain)

result = chain.invoke({
    "topic" : "Russia vs Ukraine"
})

print(result)

chain.get_graph().print_ascii()

