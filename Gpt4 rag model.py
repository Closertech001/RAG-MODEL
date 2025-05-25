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
from openai import OpenAI
from datetime import datetime
import time

# -- Safety check: API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not set in environment.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations dictionary
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
    try:
        with open("qa_dataset.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load qa_dataset.json: {e}")
        st.stop()
        
    rag_data = []
    for entry in raw_data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        department = entry.get("department", "").strip()
        level = entry.get("level", "").strip()
        semester = entry.get("semester", "").strip()
        faculty = entry.get("faculty", "").strip()

        if question and answer:
            combined_text = f"Q: {question}\nA: {answer}"
            rag_data.append({
                "text": combined_text,
                "question": question,
                "department": department,
                "level": level,
                "semester": semester,
                "faculty": faculty
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

def get_related_questions(user_query, df, index, model, top_k=6):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

# Summarize older history if > 12 messages (keeps last 6 detailed)
def summarize_history(history):
    if len(history) <= 12:
        return None, history
    to_summarize = history[:-6]
    recent = history[-6:]

    summary_prompt = [
        {"role": "system", "content": "Summarize the following conversation between user and assistant into concise bullet points for context."},
    ] + [{"role": m["role"] if m["role"]!="bot" else "assistant", "content": m["content"]} for m in to_summarize]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=summary_prompt,
            temperature=0.5,
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = None
    return summary, recent

def generate_prompt_with_memory(history, context_string, user_query, summary=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for Crescent University."}
    ]
    if summary:
        messages.append({"role": "system", "content": f"Conversation summary so far:\n{summary}"})

    if context_string:
        messages.append({"role": "system", "content": f"Context:\n{context_string}"})

    # Add recent conversation messages (last 6)
    for msg in history[-6:]:
        role = "assistant" if msg["role"] == "bot" else msg["role"]
        messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})
    return messages

def query_gpt_with_context_and_memory(user_query, df, index, model, history, top_k=6):
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

    summary, recent_history = summarize_history(history)
    messages = generate_prompt_with_memory(recent_history, context_string, user_query, summary)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip() if response.choices else "⚠️ No response generated."
    except Exception as e:
        return f"❌ GPT API Error: {str(e)}"

# --- UI Setup ---
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

df = load_data()
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

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
    
    if st.button("Clear Filters"):
        selected_faculty = "All"
        selected_department = "All"
        selected_level = "All"
        selected_semester = "All"
        st.experimental_rerun()

# Display Chat with timestamps
st.title("🎓 Crescent University Chatbot")

for chat in st.session_state.history:
    timestamp = datetime.now().strftime("%H:%M:%S")
    if chat["role"] == "user":
        st.chat_message("user").write(f"{chat['content']}  \n*{timestamp}*")
    else:
        st.chat_message("assistant").write(f"{chat['content']}  \n*{timestamp}*")

# Chat input
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)

    if not filtered_df.empty:
        filtered_embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
        index = build_faiss_index(filtered_embeddings)

        # Typing indicator
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("Typing...")
            # Simulate minimal delay for typing effect
            time.sleep(0.7)

            response = query_gpt_with_context_and_memory(user_input, filtered_df, index, model, st.session_state.history)
            placeholder.markdown(response)

        st.session_state.history.append({"role": "bot", "content": response})

        related = get_related_questions(user_input, filtered_df, index, model)
        st.session_state.related_questions = related
    else:
        st.session_state.history.append({
            "role": "bot",
            "content": "⚠️ No matching data found for the selected filters."
        })

# Related questions UI - buttons in 3 columns
if st.session_state.related_questions:
    st.subheader("🔍 Related Questions:")
    cols = st.columns(3)
    for i, q in enumerate(st.session_state.related_questions):
        if cols[i % 3].button(q):
            st.session_state.history.append({"role": "user", "content": q})
            filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)
            if not filtered_df.empty:
                filtered_embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
                index = build_faiss_index(filtered_embeddings)
                # Typing indicator for related question response
                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    placeholder.markdown("Typing...")
                    time.sleep(0.7)
                    response = query_gpt_with_context_and_memory(q, filtered_df, index, model, st.session_state.history)
                    placeholder.markdown(response)

                st.session_state.history.append({"role": "bot", "content": response})
            else:
                st.session_state.history.append({"role": "bot", "content": "⚠️ No matching data found for the selected filters."})
