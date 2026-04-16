import json
from sentence_transformers import SentenceTransformer
from endee import Client

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open("data.json") as f:
    data = json.load(f)

client = Client()

for item in data:
    embedding = model.encode(item["desc"]).tolist()

    client.insert({
        "id": item["id"],
        "vector": embedding,
        "metadata": {
            "title": item["title"],
            "desc": item["desc"]
        }
    })

print("Data inserted into Endee!")
