# --- app.py ---
import streamlit as st
import os
from rag_engine import (
    load_chunks, build_index, search, normalize_input, DEFAULT_VOCAB,
    ask_gpt_with_memory, log_feedback
)
from utils import convert_file_to_chunks, append_chunks_to_json

st.set_page_config(page_title="🎓 University Assistant", layout="wide")
st.title("🎓 Crescent University Assistant Chatbot")

# Set OpenAI Key
api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Select metadata filters
faculties = ["College of Natural and Applied Sciences", "College of Health Sciences", "College of Environmental Sciences", "Bola Ajibola College of Law", "College of Arts, Social and Management Sciences"]
departments = {
    "College of Health Sciences": ["Department of Nursing", "Department of Physiology", "Departments of Anatomy"],
    "College of Natural and Applied Sciences": ["Department of Biological Sciences(Microbiology)", "Department of Chemical Sciences(Biochemistry)", "department of Computer Science"],
    "College of Environmental Sciences": ["Department of Architecture"],
    "College of Arts, Social and Management Sciences": ["Department of Accounting", "Department of Business Administration", "Department of Economics with Operations Research", "Department of Mass Communication", "Department of Political Science and International Studies"],
    "Bola Ajibola College of Law": ["Department of Law (LL.B)"]
}

faculty = st.selectbox("Select Faculty", faculties)
department = st.selectbox("Select Department", departments[faculty])
level = st.selectbox("Select Level", ["100", "200", "300", "400", "500"])

# Load & index
@st.cache_resource
def setup():
    chunks, _ = load_chunks()
    index, model, _ = build_index([c for c in chunks])
    return chunks, index, model

chunks, index, model = setup()

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": f"You are a helpful assistant for students in the {department} department, {faculty} faculty."}
    ]

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
for msg in st.session_state.chat_history[1:]:
    who = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{who}:** {msg['content']}")

# Ask a question
query = st.text_input("💬 Ask a question")
if query and api_key:
    normalized = normalize_input(query, DEFAULT_VOCAB, model)
    top_chunks = search(normalized, index, model, [c for c in chunks], top_k=3)
    context = "\n\n".join(top_chunks)
    prompt = f"Use this context to answer as clearly as possible:\n\n{context}\n\nQuestion: {normalized}"
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    reply, usage = ask_gpt_with_memory(st.session_state.chat_history, max_history=6)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
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
