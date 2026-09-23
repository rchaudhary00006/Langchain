from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel , RunnablePassthrough, RunnableLambda

load_dotenv()

def count_words(text: str) -> int:
    return len(text.split())

count_words_runnable = RunnableLambda(count_words)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Tell me a joke on the topic {topic}",
    input_variables=['topic']
)

joke_generator_chain = RunnableSequence(prompt1, model, parser)

# word_count_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': count_words_runnable
# })

word_count_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x : len(x.split()))
})



chain = RunnableSequence(joke_generator_chain, word_count_chain)

result = chain.invoke({
    'topic' : 'Cricket'
})

print(result)

chain.get_graph().print_ascii()

