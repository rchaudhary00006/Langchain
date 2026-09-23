from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template= "Tell me a joke on the topic {topic}, the joke should be one-liner",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template= "Explain the follwing joke -> {joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)

result = chain.invoke({
    'topic' : 'Black hole'
})

print(result)