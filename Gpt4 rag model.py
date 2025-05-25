import streamlit as st
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import random

openai.api_key = os.getenv("OPENAI_API_KEY")

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between",
    "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information", "l": "level", "CSC": "Computer Science",
    "ECO": "Economics with Operations Research", "PHY": "Physics", "STAT": "Statistics",
    "1st": "First", "2nd": "Second"
}

def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word.lower(), word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you", "how are you doing",
                 "how's it going", "can we talk?", "can we have a conversation?", "okay", "i'm fine", "i am fine"]
    return text.lower().strip() in greetings

@st.cache_data
def load_data():
    with open("qa_dataset (1).json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    rag_data = []
    for entry in raw_data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        if question and answer:
            combined_text = f"Q: {question}\nA: {answer}"
            rag_data.append({"text": combined_text, "question": question})
    return pd.DataFrame(rag_data)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def query_gpt4_with_context(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    if is_greeting(clean_query):
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!", "I'm doing well, thank you!", "Sure pal", "Okay", "I'm fine, thank you"])
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    context_blocks = [df.iloc[i]['text'] for i in I[0] if i < len(df)]
    context_string = "\n\n".join(context_blocks)
    system_prompt = (
        "You are a helpful assistant for Crescent University. "
        "Use the provided information to answer the user's question accurately. "
        "If the answer is not clearly stated, say you don't know and suggest referring to official sources."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Based on the following context:\n\n{context_string}\n\nAnswer the following question:\n{user_query}"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"\u274c Error calling GPT-4: {str(e)}"

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

# Streamlit UI setup
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

st.title("🎓 Crescent University RAG Chatbot")

# Load data/model/index
df = load_data()
model = load_model()
embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)
index = build_faiss_index(embeddings)

if "history" not in st.session_state:
    st.session_state.history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Sidebar
with st.sidebar:
    st.title("🧭 Options")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.related_questions = []
        st.session_state.feedback = []
        st.experimental_rerun()

# CSS Styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans&display=swap" rel="stylesheet">
<style>
    html, body, .stApp {
        font-family: 'Open Sans', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Merriweather', serif;
        color: #004080;
    }
    .chat-container {
        max-height: 480px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #d6eaff;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: auto;
        max-width: 75%;
        font-weight: 550;
        color: #000;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-right: auto;
        max-width: 75%;
        font-weight: 600;
        color: #000;
    }
    .related-question {
        background-color: #e6f2ff;
        padding: 8px 12px;
        margin: 6px 6px 6px 0;
        display: inline-block;
        border-radius: 10px;
        font-size: 0.9rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Display Chat Messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for chat in st.session_state.history:
    role_class = "user-message" if chat["role"] == "user" else "bot-message"
    label = "You" if chat["role"] == "user" else "Bot"
    st.markdown(f'<div class="{role_class}"><strong>{label}:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input
user_input = st.text_input("Ask your question:")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    response = query_gpt4_with_context(user_input, df, index, model)
    st.session_state.history.append({"role": "bot", "content": response})
    st.session_state.related_questions = get_related_questions(user_input, df, index, model)

# Related Questions
if st.session_state.related_questions:
    st.markdown("#### 🔍 Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q):
            st.session_state.history.append({"role": "user", "content": q})
            response = query_gpt4_with_context(q, df, index, model)
            st.session_state.history.append({"role": "bot", "content": response})
