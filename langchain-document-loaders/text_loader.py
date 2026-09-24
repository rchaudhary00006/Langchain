from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

parser = StrOutputParser()

text_loader = TextLoader("cricket.txt")

docs = text_loader.load()

prompt = PromptTemplate(
    template= "Write summary for the following poem {poem}",
    input_variables=['poem']
)

get_poem = RunnableLambda(lambda x: x[0].page_content)

chain = get_poem | prompt | model | parser

result = chain.invoke(docs)

print(result)

chain.get_graph().print_ascii()