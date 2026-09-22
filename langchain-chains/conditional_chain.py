from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq
import os

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)

model = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"), temperature=0)

# model = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite"
# )

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

strparser = StrOutputParser()

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template="Give the sentiment of the feedback into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydantic_parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template="Give the appropriate response to this posiitve feedback in a para \n {feedback}",
    input_variables=['feedback']
)

template3 = PromptTemplate(
    template="Give the appropriate response  to this negative feedback in a para \n {feedback}",
    input_variables=['feedback']
)

classifier_chain = template1 | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', template2 | model | strparser),
    (lambda x: x.sentiment == 'negative', template3 | model | strparser),
    RunnableLambda(lambda x : "Could not find sentiment")
)

chain = classifier_chain | branch_chain 

result =  chain.invoke({
    "feedback" : "This is a rubbish device."
})

print(result)

chain.get_graph().print_ascii()