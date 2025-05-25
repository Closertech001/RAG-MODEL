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

# -- Safety check: API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not set in environment.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize SymSpell
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
    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?", "can we have a conversation?",
        "okay", "i'm fine", "i am fine"
    ]
    return text.lower().strip() in greetings

@st.cache_data(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource(show_spinner=False)
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

def generate_prompt_with_memory(history, context_string, user_query):
    # Limit history length to last 6 messages for token economy
    recent_history = history[-6:] if len(history) > 6 else history

    messages = [
        {"role": "system", "content": "You are a helpful assistant for Crescent University. Use the context and remember the conversation."}
    ]

    # Add context as system message for knowledge base
    if context_string:
        messages.append({"role": "system", "content": f"Context:\n{context_string}"})

    # Add conversation history
    for msg in recent_history:
        # Roles: 'user' or 'assistant'
        if msg["role"] == "bot":
            role = "assistant"
        else:
            role = msg["role"]
        messages.append({"role": role, "content": msg["content"]})

    # Add current user query last
    messages.append({"role": "user", "content": user_query})

    return messages

def query_openai_with_memory(user_query, df, index, model, history, top_k=5):
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

    messages = generate_prompt_with_memory(history, context_string, user_query)

    # Try GPT-4 Turbo, fallback to GPT-3.5 Turbo on failure
    for model_name in ["gpt-4o-mini", "gpt-3.5-turbo"]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Log error but try next model
            st.warning(f"⚠️ Model {model_name} failed: {e}")
            continue
    return "❌ Sorry, I'm unable to process your request at the moment."

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

# --- Streamlit UI ---

st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

df = load_data()
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

# Sidebar filters & clear chat button
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

# Display chat history
st.title("🎓 Crescent University Chatbot")
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        st.chat_message("assistant").write(chat["content"])

# Input area
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)

    if not filtered_df.empty:
        # Cache embeddings & index per filter combination
        cache_key = f"{selected_faculty}_{selected_department}_{selected_level}_{selected_semester}"
        if "embedding_cache" not in st.session_state:
            st.session_state.embedding_cache = {}
        if "index_cache" not in st.session_state:
            st.session_state.index_cache = {}

        if cache_key not in st.session_state.embedding_cache:
            embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
            st.session_state.embedding_cache[cache_key] = embeddings
            st.session_state.index_cache[cache_key] = build_faiss_index(embeddings)

        embeddings = st.session_state.embedding_cache[cache_key]
        index = st.session_state.index_cache[cache_key]

        # Query with memory
        response = query_openai_with_memory(user_input, filtered_df, index, model, st.session_state.history)
        st.session_state.history.append({"role": "bot", "content": response})

        # Update related questions
        related = get_related_questions(user_input, filtered_df, index, model)
        st.session_state.related_questions = related
    else:
        st.session_state.history.append({
            "role": "bot",
            "content": "⚠️ No matching data found for the selected filters."
        })

# Related questions buttons
if st.session_state.related_questions:
    st.subheader("🔍 Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q):
            st.session_state.history.append({"role": "user", "content": q})
            filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)
            if not filtered_df.empty:
                cache_key = f"{selected_faculty}_{selected_department}_{selected_level}_{selected_semester}"
                embeddings = st.session_state.embedding_cache.get(cache_key) or model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
                index = st.session_state.index_cache.get(cache_key) or build_faiss_index(embeddings)

                response = query_openai_with_memory(q, filtered_df, index, model, st.session_state.history)
                st.session_state.history.append({"role": "bot", "content": response})

