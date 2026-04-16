from sentence_transformers import SentenceTransformer
from endee import Client

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Client()

query = input("Enter what you like: ")

query_embedding = model.encode(query).tolist()

results = client.search(query_embedding, top_k=3)

print("\nRecommended Movies:\n")

for r in results:
    print(r["metadata"]["title"])
