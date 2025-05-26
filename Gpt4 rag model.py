import streamlit as st
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import random
import openai

# Page config must be first!
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

# Custom Styling
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
    .chat-message-user {
        background-color: #d6eaff;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: auto;
        max-width: 75%;
        font-weight: 550;
        color: #000;
    }
    .chat-message-assistant {
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

# API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# SymSpell Setup
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

abbreviations = {
    "u": "you", "r": "are", "ur": "your", "pls": "please", "tmrw": "tomorrow",
    "cn": "can", "wat": "what", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "btw": "between", "idk": "i don't know",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information", "l": "level"
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
    greetings = ["hi", "hello", "hey", "hi there", "how are you", "greetings"]
    return text.lower().strip() in greetings

@st.cache_data
def load_data():
    try:
        with open("qa_dataset.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load qa_dataset.json: {e}")
        st.stop()

    rag_data = []
    for entry in raw_data:
        rag_data.append({
            "text": f"Q: {entry['question']}\nA: {entry['answer']}",
            **entry
        })
    return pd.DataFrame(rag_data)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def fallback_openai(user_input, context_qa=None):
    system_prompt = "You are a helpful assistant specialized in Crescent University."
    messages = [{"role": "system", "content": system_prompt}]
    if context_qa:
        messages.append({"role": "user", "content": f"{context_qa['question']}\n{context_qa['answer']}\n\nAnswer: {user_input}"})
    else:
        messages.append({"role": "user", "content": user_input})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return "Sorry, I couldn't reach the server. Try again later."

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

def apply_filters(df, faculty, department, level, semester):
    if faculty != "All":
        df = df[df["faculty"] == faculty]
    if department != "All":
        df = df[df["department"] == department]
    if level != "All":
        df = df[df["level"] == level]
    if semester != "All":
        df = df[df["semester"] == semester]
    return df

# Load
df = load_data()
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}
if "faiss_index_cache" not in st.session_state:
    st.session_state.faiss_index_cache = {}

# Sidebar
with st.sidebar:
    st.title("Crescent University RAG Chatbot")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.related_questions = []
        st.experimental_rerun()

    st.subheader("📚 Filters")
    selected_faculty = st.selectbox("Faculty", ["All"] + sorted(df["faculty"].dropna().unique().tolist()))
    selected_department = st.selectbox("Department", ["All"] + sorted(df["department"].dropna().unique().tolist()))
    selected_level = st.selectbox("Level", ["All"] + sorted(df["level"].dropna().unique().tolist()))
    selected_semester = st.selectbox("Semester", ["All"] + sorted(df["semester"].dropna().unique().tolist()))

# Main
st.title("🎓 Crescent University Chatbot")

chat_placeholder = st.container()

with chat_placeholder:
    for chat in st.session_state.history:
        class_name = "chat-message-user" if chat["role"] == "user" else "chat-message-assistant"
        st.markdown(f"<div class='{class_name}'>{chat['content']}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    if is_greeting(user_input):
        reply = random.choice(["Hi! How can I help you?", "Hello! Ask me anything about Crescent University."])
    else:
        filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)
        cache_key = (selected_faculty, selected_department, selected_level, selected_semester)

        if cache_key not in st.session_state.embedding_cache:
            embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
            st.session_state.embedding_cache[cache_key] = embeddings
            st.session_state.faiss_index_cache[cache_key] = build_faiss_index(embeddings)

        index = st.session_state.faiss_index_cache[cache_key]
        query_embedding = model.encode([preprocess_text(user_input)], convert_to_numpy=True)
        D, I = index.search(query_embedding, 1)
        best_idx = I[0][0]
        context_qa = {
            "question": filtered_df.iloc[best_idx]["question"],
            "answer": filtered_df.iloc[best_idx]["answer"]
        }
        reply = fallback_openai(user_input, context_qa)

        st.session_state.related_questions = get_related_questions(user_input, filtered_df, index, model)

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.rerun()

# Related
if st.session_state.related_questions:
    st.subheader("🔍 Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q):
            st.session_state.history.append({"role": "user", "content": q})
            st.rerun()
