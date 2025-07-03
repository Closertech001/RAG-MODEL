# app.py
import streamlit as st
import os
import time
import openai
from rag_engine import load_chunks, build_index, search, normalize_input, ask_gpt_with_memory, is_small_talk, handle_small_talk
from db import init_tables, get_user, create_user, save_chat
from memory import ContextMemory
from symspell_setup import correct_query
from config import BOT_PERSONALITY, SMALL_TALK_PATTERNS
from sentence_transformers import SentenceTransformer
import random

# --- App Config ---
st.set_page_config(page_title="🎓 CrescentBot AI Assistant", layout="wide")
st.title("🤖 Crescent University Assistant Bot")

# --- Init DB Tables ---
init_tables()

# --- Load Data and Build Index ---
with st.spinner("🔍 Initializing engine..."):
    chunks, raw_data = load_chunks("qa_dataset.json")
    index, model, embeddings = build_index(chunks)

# --- Session State Init ---
if "context" not in st.session_state:
    st.session_state.context = ContextMemory()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# --- Sidebar: User Auth ---
st.sidebar.header("🔐 User Profile")
name = st.sidebar.text_input("Enter your name")
faculty = st.sidebar.selectbox("Faculty", ["", "Health Sciences", "ICT", "Law", "Natural Sciences", "Social Sciences"])
department = st.sidebar.text_input("Department")

if name and st.sidebar.button("Start Chat"):
    user = get_user(name)
    if not user:
        st.session_state.user_id = create_user(name, faculty, department)
    else:
        st.session_state.user_id = user["id"]
    st.success(f"Welcome, {name}! 👋")

# --- Chat Interface ---
st.markdown("---")
st.markdown(BOT_PERSONALITY["greeting"])
user_query = st.text_input("Ask me anything about Crescent University:")

if user_query:
    with st.spinner("Thinking..."):
        corrected = correct_query(user_query)

        if is_small_talk(corrected):
            response = handle_small_talk(corrected)
        else:
            norm_query = normalize_input(corrected, None, model)
            top_match, score = search(norm_query, index, model, chunks, top_k=1)

            # If confidence is low, use GPT
            if score > 0.6:
                response = top_match[0]
            else:
                history = st.session_state.context.get_context()
                messages = [{"role": "system", "content": f"You are {BOT_PERSONALITY['name']}, an academic assistant."}]
                for turn in history:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["bot"]})
                messages.append({"role": "user", "content": user_query})
                response, _ = ask_gpt_with_memory(messages)

        st.markdown(f"**You:** {user_query}")
        st.markdown(f"**{BOT_PERSONALITY['name']}:** {response}")

        # Save to DB and memory
        if st.session_state.user_id:
            save_chat(st.session_state.user_id, user_query, response)
        st.session_state.context.add_turn(user_query, response)

        # Optional rating
        rating = st.radio("Was this helpful?", ["👍", "👎"], horizontal=True, key=f"rate-{random.randint(1,10000)}")
        if rating:
            with open("feedback.csv", "a") as f:
                f.write(f"{time.time()},\"{user_query}\",\"{response}\",{rating}\n")

# --- Upload Docs ---
st.sidebar.markdown("---")
st.sidebar.header("📤 Upload University Docs")
uploaded = st.sidebar.file_uploader("Upload PDF/DOCX/CSV", type=["pdf", "docx", "txt", "csv"])

if uploaded and st.sidebar.button("Ingest File"):
    from utils import convert_file_to_chunks, append_chunks_to_json
    new_chunks = convert_file_to_chunks(uploaded, faculty, department, level="100")
    append_chunks_to_json(new_chunks)
    st.sidebar.success("File ingested! Please restart app to reindex.")
