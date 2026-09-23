from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Give me content for tweet for the topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Give me content for LinkedIn post for the topic {topic}",
    input_variables=['topic']
)

chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({
    'topic' : 'AI'
})

print("Tweet:", result['tweet'])
print()
print("LinkedIn Post:", result['linkedin'])  

chain.get_graph().print_ascii()