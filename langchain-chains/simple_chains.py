from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="Give 5 intresting fact about {topic}, all fact should be one-liner",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({
    'topic' : 'Black hole'
})

print(result)

chain.get_graph().print_ascii()