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
    df = pd.DataFrame(rag_data)
    return df

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
    context_blocks = []
    for i in I[0]:
        if i < len(df):
            context_blocks.append(df.iloc[i]['text'])
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
        return f"❌ Error calling GPT-4: {str(e)}"

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    related_qs = []
    for i in I[0]:
        if i < len(df):
            related_qs.append(df.iloc[i]['question'])
    return related_qs

# --- Streamlit UI setup ---

st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

# Load data and model once
df = load_data()
model = load_model()
embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)
index = build_faiss_index(embeddings)

# Initialize session state for chat history, related questions, and feedback
if "history" not in st.session_state:
    st.session_state.history = []

if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# CSS styles for chat UI and related questions
st.markdown("""
<style>
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
    background-color: #DCF8C6;
    padding: 12px 15px;
    border-radius: 15px 15px 0 15px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 10px;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    font-size: 15px;
    word-wrap: break-word;
}
.bot-message {
    background-color: #F1F0F0;
    padding: 12px 15px;
    border-radius: 15px 15px 15px 0;
    max-width: 70%;
    margin-right: auto;
    margin-bottom: 10px;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    font-size: 15px;
    word-wrap: break-word;
}
.related-container {
    margin-top: 10px;
    margin-bottom: 20px;
}
.related-button {
    background-color: #e0e0e0;
    border: none;
    border-radius: 20px;
    padding: 7px 15px;
    margin: 5px 5px 5px 0;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s ease;
}
.related-button:hover {
    background-color: #a0c4ff;
    color: #000;
}
.feedback-buttons button {
    margin-right: 10px;
}
.input-container {
    display: flex;
    margin-top: 15px;
}
input[type="text"] {
    flex-grow: 1;
    padding: 10px;
    font-size: 16px;
    border-radius: 8px 0 0 8px;
    border: 1px solid #ccc;
    outline: none;
}
button.send-btn {
    padding: 10px 20px;
    font-size: 16px;
    border-radius: 0 8px 8px 0;
    border: 1px solid #ccc;
    background-color: #4CAF50;
    color: white;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
button.send-btn:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# --- Display chat history ---
def display_messages():
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for i, chat in enumerate(st.session_state.history):
        if chat["role"] == "user":
            st.markdown(
                f'<div class="user-message"><strong>You:</strong> {chat["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bot-message"><strong>Bot:</strong> {chat["content"]}</div>',
                unsafe_allow_html=True)
            cols = st.columns([1,1,8])
            with cols[0]:
                if st.button("👍", key=f"like_{i}"):
                    st.session_state.feedback.append({"index": i, "feedback": "like"})
            with cols[1]:
                if st.button("👎", key=f"dislike_{i}"):
                    st.session_state.feedback.append({"index": i, "feedback": "dislike"})
    st.markdown('</div>', unsafe_allow_html=True)

# --- Display related questions ---
def display_related_questions():
    if st.session_state.related_questions:
        st.markdown('<div class="related-container"><strong>🔍 Related Questions:</strong></div>', unsafe_allow_html=True)
        # Show buttons in a horizontal layout with wrapping
        cols = st.columns(len(st.session_state.related_questions))
for idx, rq in enumerate(st.session_state.related_questions):
    with cols[idx]:
        if st.button(rq, key=f"related_{idx}"):
            st.session_state.history.append({"role": "user", "content": rq})
            response = query_gpt4_with_context(rq, df, index, model)
            st.session_state.history.append({"role": "bot", "content": response})
            st.session_state.related_questions = get_related_questions(rq, df, index, model)
            st.experimental_rerun()

