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
import uuid
from openai import OpenAI

# --- OpenAI API Key Check ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("🚫 OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --- Initialize OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Initialize SymSpell ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- Abbreviation Dictionary ---
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

# --- Preprocessing ---
def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you", "how are you doing",
                 "how's it going", "can we talk?", "can we have a conversation?", "okay", "i'm fine", "i am fine"]
    return text.lower().strip() in greetings

# --- Load and Prepare Dataset ---
@st.cache_data
def load_data():
    with open("qa_dataset (1).json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    rag_data = []
    for entry in raw_data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        department = entry.get("department", "").strip()
        level = entry.get("level", "").strip()
        if question and answer:
            combined_text = f"Q: {question}\nA: {answer}"
            rag_data.append({
                "text": combined_text,
                "question": question,
                "department": department,
                "level": level
            })
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

@st.cache_data
def get_embeddings(df, model):
    return model.encode(df["text"].tolist(), convert_to_numpy=True)

def query_gpt4_with_context(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    if is_greeting(clean_query):
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!",
            "I'm doing well, thank you!", "Sure pal", "Okay", "I'm fine, thank you"
        ])
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error calling GPT-4: {str(e)}"

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

st.title("Crescent University Chatbot 👨‍🎓")

# --- Load and Filter Data ---
df = load_data()
model = load_model()

# Sidebar filters
with st.sidebar:
    st.title("🔍 Filter Options")
    departments = sorted(df["department"].dropna().unique())
    levels = sorted(df["level"].dropna().unique())
    selected_department = st.selectbox("📘 Department", ["All"] + departments)
    selected_level = st.selectbox("🎓 Level", ["All"] + levels)
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.related_questions = []
        st.session_state.feedback = []
        st.experimental_rerun()

# Filter dataframe
filtered_df = df.copy()
if selected_department != "All":
    filtered_df = filtered_df[filtered_df["department"] == selected_department]
if selected_level != "All":
    filtered_df = filtered_df[filtered_df["level"] == selected_level]

if len(filtered_df) == 0:
    st.warning("⚠️ No questions found for the selected filters.")
    st.stop()

# Load embeddings and FAISS index
embeddings = get_embeddings(filtered_df, model)
index = build_faiss_index(embeddings)

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# --- Display Chat ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for i, chat in enumerate(st.session_state.history):
    role_class = "user-message" if chat["role"] == "user" else "bot-message"
    label = "You" if chat["role"] == "user" else "Bot"
    st.markdown(f'<div class="{role_class}"><strong>{label}:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
    if chat["role"] == "bot":
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍", key=f"up_{i}"):
                st.session_state.feedback.append({"question": st.session_state.history[i - 1]["content"], "feedback": "positive"})
        with col2:
            if st.button("👎", key=f"down_{i}"):
                st.session_state.feedback.append({"question": st.session_state.history[i - 1]["content"], "feedback": "negative"})
st.markdown('</div>', unsafe_allow_html=True)

# --- User Input ---
user_input = st.text_input("Ask your question:")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    response = query_gpt4_with_context(user_input, filtered_df, index, model)
    st.session_state.history.append({"role": "bot", "content": response})
    st.session_state.related_questions = get_related_questions(user_input, filtered_df, index, model)

# --- Related Questions ---
if st.session_state.related_questions:
    st.markdown("#### 🔍 Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q, key=f"related_{uuid.uuid4()}"):
            st.session_state.history.append({"role": "user", "content": q})
            response = query_gpt4_with_context(q, filtered_df, index, model)
            st.session_state.history.append({"role": "bot", "content": response})
