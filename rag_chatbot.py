import os
import requests
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Did you create .env containing a key?")

BASE_URL = "https://api.cohere.com"
HEADERS = {
    "Authorization": f'Bearer {API_KEY}',
    "Content-Type": "application/json",
}

# read lines of knowledge base, replace characters, and strip whitespace
knowledge_base = []
with open('./knowledge.txt') as knowledge_in:
    for line in knowledge_in:
        knowledge_base.append(line.replace('\u201d','"').replace('\u201c','"').replace('\u2019',"'").rstrip())
# lines stored in list

# get k most relevant knowledge base entries to the query
def retrieve_context(query, top_k=3):
    url = f"{BASE_URL}/v2/rerank" # Ranks texts by relevance to input query
    payload = {
        "model":"rerank-v3.5",
        "query":query,
        "documents":knowledge_base,
        "top_n":top_k
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    greatest_relevance = 0
    context = []
    if response.status_code == 200:
        data = response.json()
        results = data.get("results")
        if results:
            k_entries = results[0:top_k]
            greatest_relevance = k_entries[0]["relevance_score"]
            for entry in k_entries:
                context.append(knowledge_base[entry["index"]])
        #most_relevant = data.get("results", [{"relevance_score":0, "index":0}])[0]
        #print(f'Relevance score {most_relevant["relevance_score"]:.3f}:\n   {payload["documents"][most_relevant["index"]]}')
    else:
        print(f"[ERROR] {response.status_code}: {response.text}")
    #print((greatest_relevance, context))
    return (greatest_relevance, context)

def system_msg(context):
    return f'''
You are to advise the user on computer parts and what they should know about each component. 
Use only the following context to answer the question. In the answer, do not mention that this context was provided. 
Refrain from adding information that was not included in the context. 
If the context does not include the answer to the user's question, respond with: 
"I don't know based on my knowledge base."

Context:
{context}
'''

# prompt to be fed to the response generator
# context is the knowledge base entries from retrieve_context
def build_prompt(context, question):
    return [
        {"role":"system", "content":system_msg(context)},
        {"role":"user", "content": question}
    ]

# for response generation
# MODEL_NAME = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(prompt: list, max_new_tokens: int = 80) -> str:
    url = f'{BASE_URL}/v2/chat'
    payload = {
        "stream":False,
        "model":"command-a-03-2025",
        "messages":prompt,
        "max_tokens":max_new_tokens
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("message").get("content", [{}])[0].get("text")
    else:
        return f"[ERROR] {response.status_code}: {response.text}"

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
    if context[0] > 0.45:
        response = generate_answer(build_prompt(context[1], query), max_new_tokens=480)
    else:
        response = "I don't know based on my knowledge base."


with st.chat_message("assistant"):
    st.markdown(response)

st.session_state.messages.append({"role":"assistant", "content":response})