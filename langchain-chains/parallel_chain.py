from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model1 = ChatHuggingFace(llm=llm1)

model2 = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

template1 = PromptTemplate(
    template="Generate a notes on the following text: \n {text}",
    input_variables=['text']
)

template2 = PromptTemplate(
    template="Generate 5 quiz on the following text: \n {text}",
    input_variables=['text']
)

template3 = PromptTemplate(
    template="Merge the following notes and quiz in a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : template1 | model1 | parser,
    'quiz' : template2 | model2 | parser
})

sequential_chain = template3 | model2 | parser

chain = parallel_chain | sequential_chain

result = chain.invoke({
    'text' : """Virat Kohli is one of the most celebrated and influential cricketers in Indian cricket. He was born on 5 November 1988 in Delhi, India, and developed a passion for cricket from a young age. Kohli rose to prominence after leading the Indian Under-19 team to victory in the 2008 Under-19 Cricket World Cup. Soon after, he made his international debut for India and established himself as a dependable and highly competitive batsman. Known for his excellent technique, aggressive style, and ability to perform under pressure, Kohli has scored thousands of runs across international cricket and has achieved numerous batting records. He has been particularly successful in One Day Internationals and is widely known for his consistency while chasing targets. Kohli has also served as captain of the Indian cricket team and played an important role in developing a strong fitness culture within Indian cricket. His dedication to fitness, discipline, and continuous improvement has made him an inspiration to many young athletes. In the Indian Premier League, he has been a long-standing and prominent player for Royal Challengers Bengaluru. His passion and emotional connection with the game have made him a popular figure among cricket fans around the world. Although cricket has been the central part of his career, Kohli has also been involved in business and philanthropic activities. His journey from a young cricketer in Delhi to an internationally recognized sports personality demonstrates the importance of hard work, discipline, perseverance, and self-belief."""
})

print(result)

chain.get_graph().print_ascii()

