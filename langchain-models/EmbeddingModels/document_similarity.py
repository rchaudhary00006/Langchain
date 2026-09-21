from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Virat Kohli is one of India's greatest batsmen, known for his consistency and aggressive batting style.",
    "Rohit Sharma is an accomplished Indian batsman and former captain, famous for his elegant stroke play and record-breaking ODI performances.",
    "MS Dhoni is a legendary Indian wicketkeeper-batsman and captain, known for his calm leadership and finishing ability.",
    "Sachin Tendulkar is a cricket legend and former Indian batsman, widely regarded as one of the greatest players in the history of the sport.",
    "Jasprit Bumrah is an Indian fast bowler known for his unique bowling action, accuracy, and effectiveness in all formats.",
    "Prajval lodu hai",
    "Prajval is good hai",

]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_vectors = embeddings.embed_documents(documents)
# print("Embedding vectors generated for the documents.", embedding_vectors)
# for i , doc in enumerate(documents):
#     print(f"Document {i+1}: {doc}")
#     print(f"Embedding Vector: {embedding_vectors[i]}")
#     print()

# with open("example.txt", "w") as f:
#     for i, doc in enumerate(documents):
#         f.write(f"Document {i+1}: {doc}\n")
#         f.write(f"Embedding Vector: {embedding_vectors[i]}\n\n")
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    query_embedding = embeddings.embed_query(query)
    # cosine_similarity ke andar hamesa 2D array pass karna hota hai, isliye query_embedding ko list me wrap kiya gaya hai
    similarities = cosine_similarity([query_embedding], embedding_vectors)[0]

    print(f"\nSimilarity scores: {similarities}\n")
    most_similar_index = np.argmax(similarities) # returns the index (position) of the largest value in an array.
    most_similar_document = documents[most_similar_index]
    similarity_score = similarities[most_similar_index]

    print(f"\nMost similar document: {most_similar_document}")
    print(f"Similarity score: {similarity_score}\n")