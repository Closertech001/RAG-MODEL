# --- app.py ---
import streamlit as st
import os
import time
from rag_engine import (
    load_chunks, build_index, search, normalize_input, DEFAULT_VOCAB,
    ask_gpt_with_memory, log_feedback, is_small_talk, handle_small_talk
)
from utils import convert_file_to_chunks, append_chunks_to_json
from db import init_db, get_user, add_user, get_chat_history, save_chat, update_user_prefs, get_user_profile
from textblob import TextBlob
import openai

# Initialize database
init_db()

st.set_page_config(page_title="🎓 University Assistant", layout="wide")
st.title("🎓 Crescent University Assistant Chatbot")

# Set OpenAI Key
api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Select metadata filters
faculties = ["College of Natural and Applied Sciences", "College of Health Sciences", "College of Environmental Sciences", "Bola Ajibola College of Law", "College of Arts, Social and Management Sciences"]
departments = {
    "College of Health Sciences": ["Department of Nursing", "Department of Physiology", "Departments of Anatomy"],
    "College of Natural and Applied Sciences": ["Department of Biological Sciences(Microbiology)", "Department of Chemical Sciences(Biochemistry)", "department of Computer Science"],
    "College of Environmental Sciences": ["Department of Architecture"],
    "College of Arts, Social and Management Sciences": ["Department of Accounting", "Department of Business Administration", "Department of Economics with Operations Research", "Department of Mass Communication", "Department of Political Science and International Studies"],
    "Bola Ajibola College of Law": ["Department of Law (LL.B)"]
}

# Identify or register user
user_name = st.text_input("👤 Enter Your Name")
user_id = None
profile = {}
if user_name:
    user_id = get_user(user_name)
    if not user_id:
        faculty = st.selectbox("Select Faculty", faculties)
        department = st.selectbox("Select Department", departments[faculty])
        level = st.selectbox("Select Level", ["100", "200", "300", "400", "500"])
        user_id = add_user(user_name, department, faculty)
        update_user_prefs(user_id, level, "neutral")
        st.success(f"👋 Welcome, {user_name}!")
    else:
        profile = get_user_profile(user_id)
        department = profile.get("department", "")
        faculty = profile.get("faculty", "")
        level = profile.get("level", "")
        st.info(f"👋 Welcome back, {user_name}! You're in {department}, Level {level}.")

# Load & index
@st.cache_resource
def setup():
    chunks, _ = load_chunks()
    index, model, _ = build_index([c for c in chunks])
    return chunks, index, model

chunks, index, model = setup()

# Chat memory
if "chat_history" not in st.session_state and user_id:
    history = get_chat_history(user_id)
    if not history:
        history = [{"role": "system", "content": f"You are a helpful assistant for students like {user_name} in the {department} department, {faculty} faculty."}]
    st.session_state.chat_history = history

# Reset chat
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = st.session_state.chat_history[:1]
    st.rerun()

# Upload documents
with st.expander("📤 Upload University Docs"):
    uploaded_file = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt", "csv"])
    if uploaded_file and st.button("➕ Add to Knowledge Base"):
        new_chunks = convert_file_to_chunks(uploaded_file, faculty, department, level)
        append_chunks_to_json(new_chunks)
        st.success("Content added! Please rerun the app to reindex.")

# Display past messages
if "chat_history" in st.session_state:
    for msg in st.session_state.chat_history[1:]:
        who = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{who}:** {msg['content']}")

# Ask a question
query = st.text_input("💬 Ask a question")
if query and api_key and user_id:
    # Small talk detection
    if is_small_talk(query):
        reply = handle_small_talk(query)
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_chat(user_id, "user", query)
        save_chat(user_id, "assistant", reply)
        st.rerun()

    # Sentiment detection
    polarity = TextBlob(query).sentiment.polarity
    if polarity < -0.3:
        tone = "empathetic"
    elif polarity > 0.3:
        tone = "enthusiastic"
    else:
        tone = "neutral"
    update_user_prefs(user_id, level, tone)

    normalized = normalize_input(query, DEFAULT_VOCAB, model)
    top_chunks = search(normalized, index, model, [c for c in chunks], top_k=3)
    context = "\n\n".join(top_chunks)

    # Clarification handling
    if len(top_chunks) == 0 or len(context.strip()) < 20:
        reply = f"🤔 Hmm, I’m not sure I understand that, {user_name}. Can you clarify a bit more?"
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_chat(user_id, "user", query)
        save_chat(user_id, "assistant", reply)
        st.rerun()

    prompt = f"Use this context to answer in a {tone} tone. Refer to the student as {user_name}:

{context}

Question from {user_name}: {normalized}"
    st.session_state.chat_history.append({"role": "user", "content": query})
    save_chat(user_id, "user", query)

    # Streaming GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.chat_history + [{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.7,
        max_tokens=500
    )

    streamed_reply = ""
    response_placeholder = st.empty()
    for chunk in response:
        if "choices" in chunk and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.get("content", "")
            streamed_reply += delta
            response_placeholder.markdown(f"**🤖 Bot:** {streamed_reply}")
            time.sleep(0.01)

    st.session_state.chat_history.append({"role": "assistant", "content": streamed_reply})
    save_chat(user_id, "assistant", streamed_reply)
    st.rerun()

# Feedback
st.markdown("---")
st.subheader("📝 Feedback")
if st.session_state.get("chat_history") and len(st.session_state.chat_history) > 2:
    last_user = st.session_state.chat_history[-2]["content"]
    last_bot = st.session_state.chat_history[-1]["content"]
    col1, col2 = st.columns(2)
    if col1.button("👍 Helpful"):
        log_feedback(last_user, last_bot, "positive")
        st.success("Thanks for your feedback!")
    if col2.button("👎 Not Helpful"):
        log_feedback(last_user, last_bot, "negative")
        st.info("Feedback noted.")
