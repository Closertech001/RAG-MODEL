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

# --- Streamlit UI ---

st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓")

st.title("🎓 Crescent University RAG Chatbot with GPT-4")

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

def display_messages():
    for i, chat in enumerate(st.session_state.history):
        if chat["role"] == "user":
            st.markdown(f'<div style="text-align: right; background-color:#DCF8C6; padding:8px; border-radius:10px; margin:5px;">**You:** {chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align: left; background-color:#F1F0F0; padding:8px; border-radius:10px; margin:5px;">**Bot:** {chat["content"]}</div>', unsafe_allow_html=True)
            # Feedback buttons after bot reply
            cols = st.columns([1,1,8])
            if cols[0].button("👍", key=f"like_{i}"):
                st.session_state.feedback.append({"index": i, "feedback": "like"})
            if cols[1].button("👎", key=f"dislike_{i}"):
                st.session_state.feedback.append({"index": i, "feedback": "dislike"})

def submit():
    user_question = st.session_state.input_text.strip()
    if user_question:
        st.session_state.history.append({"role": "user", "content": user_question})
        with st.spinner("Generating answer..."):
            answer = query_gpt4_with_context(user_question, df, index, model)
        st.session_state.history.append({"role": "bot", "content": answer})
        # Get related questions except the current one
        related_questions = get_related_questions(user_question, df, index, model, top_k=5)
        st.session_state.related_questions = [rq for rq in related_questions if rq != user_question]
        st.session_state.input_text = ""

display_messages()

if st.session_state.related_questions:
    st.markdown("### 🔍 Related Questions:")
    for rq in st.session_state.related_questions:
        if st.button(rq, key=f"related_{rq}"):
            # Set input text and trigger submit for the clicked related question
            st.session_state.input_text = rq
            submit()
            # Redisplay updated messages and clear related questions (optional)
            display_messages()
            st.experimental_rerun()

# Input area with multiline + enter to send
st.text_area("Ask a question about Crescent University:", key="input_text", height=70, on_change=submit)

# Optional: show collected feedback
if st.session_state.feedback:
    st.write("Thank you for your feedback! 👍👎")
    # You can extend this to log or process feedback further
