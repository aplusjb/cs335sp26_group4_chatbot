import os
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from huggingface_hub import login
API_KEY = os.getenv("HF_KEY") # some models require a huggingface access token
if not API_KEY:
    raise ValueError("API key not found. Did you create .env containing a key?")
login(API_KEY)

# read lines of knowledge base, replace characters, and strip whitespace
knowledge_base = []
with open('./knowledge.txt') as knowledge_in:
    for line in knowledge_in:
        knowledge_base.append(line.replace('\u201d','"').replace('\u201c','"').replace('\u2019',"'").rstrip())
# lines stored in list

embedder = SentenceTransformer("google/embeddinggemma-300m")

# embeddings are stored in vector database for fast similarity search
client = chromadb.Client()
collection = client.get_or_create_collection("rag_knowledge_base")
collection.add(
    documents=knowledge_base,
    ids=[str(uuid.uuid4()) for _ in knowledge_base],
    embeddings=embedder.encode(knowledge_base)
)

# get k most relevant knowledge base entries to the query
def retrieve_context(query, top_k=3):
    # embed query
    query_embedding = embedder.encode([query]).tolist()
    # query database with that embedding
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]

# prompt to be fed to the response generator
# context is the knowledge base entries from retrieve_context
def build_prompt(context, question):
    return f'''
Use the following context to answer the question.

Context:
{context}

Question:
{question}
'''

# for response generation
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(prompt: str, max_new_tokens: int = 80) -> str:
    """Generate an answer from a seq2seq model (deterministic)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # deterministic
            num_beams=1
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# query = "What are the sizes of motherboard?"
# context = retrieve_context(query, top_k=5) # top 5 entries seems to work better than 3
# generate_answer(build_prompt(context, query))

st.title("PC Parts Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

response = "Placeholder"

if query := st.chat_input("Ask me about computers!"):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})
    context = retrieve_context(query, top_k=5)
    response = generate_answer(build_prompt(context, query))


with st.chat_message("assistant"):
    st.markdown(response)

st.session_state.messages.append({"role":"assistant", "content":response})