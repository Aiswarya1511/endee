import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Client

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Client()

st.title("🎬 Movie Recommendation System")

query = st.text_input("What do you like?")

if query:
    query_embedding = model.encode(query).tolist()
    results = client.search(query_embedding, top_k=3)

    st.write("### Recommended Movies:")
    for r in results:
        st.write(r["metadata"]["title"])
