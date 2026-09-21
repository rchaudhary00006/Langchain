from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "This is a sample text to generate embeddings."
# embedding_vector = embeddings.embed_query(text)
documents = [
    "This is the first document.",
    "This is the second document.",
    "This is the third document.",
    "This is the fourth document."
]

embedding_vector = embeddings.embed_documents(documents)

print(embedding_vector)

