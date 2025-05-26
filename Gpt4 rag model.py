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

# PAGE CONFIG
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

# Inject CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans&display=swap" rel="stylesheet">
<style>
    html, body, .stApp { font-family: 'Open Sans', sans-serif; }
    h1, h2, h3, h4, h5 { font-family: 'Merriweather', serif; color: #004080; }
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

# API KEY
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Spell Correction Setup
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
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    rag_data = [
        {"text": f"Q: {entry['question']}\nA: {entry['answer']}", **entry} for entry in raw_data
    ]
    return pd.DataFrame(rag_data)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def fallback_openai(user_input, context_qa=None, model_name="gpt-4"):
    system_prompt = "You are a helpful assistant specialized in Crescent University information. Be concise and factual."
    messages = [{"role": "system", "content": system_prompt}]
    history = st.session_state.history[-10:]
    for chat in history:
        messages.append({"role": chat["role"], "content": chat["content"]})
    if context_qa:
        messages.append({
            "role": "user",
            "content": f"Use this info:\nQ: {context_qa['question']}\nA: {context_qa['answer']}"
        })
    messages.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.3
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

# Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}
if "faiss_index_cache" not in st.session_state:
    st.session_state.faiss_index_cache = {}

# Load data/model
df = load_data()
model = load_model()

# Sidebar
with st.sidebar:
    st.title("Crescent University RAG Chatbot")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.related_questions = []
    st.subheader("📚 Filters")
    selected_faculty = st.selectbox("Faculty", ["All"] + sorted(df["faculty"].dropna().unique().tolist()))
    selected_department = st.selectbox("Department", ["All"] + sorted(df["department"].dropna().unique().tolist()))
    selected_level = st.selectbox("Level", ["All"] + sorted(df["level"].dropna().unique().tolist()))
    selected_semester = st.selectbox("Semester", ["All"] + sorted(df["semester"].dropna().unique().tolist()))
    st.subheader("⚙️ Model")
    model_choice = st.radio("Choose model", ["gpt-3.5-turbo", "gpt-4"], index=1)

# Main title
st.title("🎓 Crescent University Chatbot")
st.markdown(f"**Model in use:** `{model_choice}`")
chat_container = st.container()

def render_chat():
    with chat_container:
        for chat in st.session_state.history:
            cls = "chat-message-user" if chat["role"] == "user" else "chat-message-assistant"
            st.markdown(f"<div class='{cls}'>{chat['content']}</div>", unsafe_allow_html=True)
        st.markdown("<div id='scroll-target'></div>", unsafe_allow_html=True)
        st.markdown("""
            <script>
                const element = document.getElementById('scroll-target');
                if (element) {
                    element.scrollIntoView({behavior: 'smooth'});
                }
            </script>
        """, unsafe_allow_html=True)

render_chat()

# Chat input
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    if is_greeting(user_input):
        reply = random.choice([
            "Hi! How can I help you?",
            "Hello! Ask me anything about Crescent University."
        ])
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

        reply = fallback_openai(user_input, context_qa, model_choice)
        st.session_state.related_questions = get_related_questions(user_input, filtered_df, index, model)

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.stop()

# Show related questions
if st.session_state.related_questions:
    st.subheader("🔍 Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q):
            st.session_state.history.append({"role": "user", "content": q})
            st.stop()
