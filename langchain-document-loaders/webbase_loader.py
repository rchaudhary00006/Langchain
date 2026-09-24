from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

url = "https://www.amazon.in/realme-Earbuds-Drivers-Playtime-Chocolate/dp/B0GVZ7JJRP?th=1"

url_loader = WebBaseLoader(url)  # for multiple urls we can also pass a list of urls and it will return a document for each url

docs = url_loader.load()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=['question', 'text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "question" : "What is the name and price of the product and also give me it's ratings ?",
    "text" : docs[0].page_content
})

print(result)