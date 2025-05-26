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

# --- API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- SymSpell Setup ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- Abbreviations Dictionary ---
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between",
    "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information", "l": "level", 
    "CSC": "Computer Science", "ECO": "Economics with Operations Research", 
    "PHY": "Physics", "STAT": "Statistics", "1st": "First", "2nd": "Second"
}

# --- Text normalization and preprocessing ---
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
        "hi", "hello", "hey", "hi there", "greetings", "how are you", "how are you doing",
        "how's it going", "can we talk?", "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    return text.lower().strip() in greetings

def normalize_filter_value(value):
    if value == "All":
        return "All"
    return value.strip()

# --- Data Loading ---
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
                "answer": answer,
                "department": department,
                "level": level,
                "semester": semester,
                "faculty": faculty
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

def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "If you don't know an answer, politely say so and refer to university resources."
    )
    messages = [{"role": "system", "content": system_prompt}]

    if context_qa:
        context_text = f"Here is some relevant university information:\nQ: {context_qa['question']}\nA: {context_qa['answer']}\n\n"
        user_message = context_text + "Answer this question: " + user_input
    else:
        user_message = user_input

    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return "Sorry, I couldn't reach the server. Try again later."

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

# --- Streamlit App ---
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="🎓", layout="wide")

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

# --- Sidebar ---
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

# --- Main Chat Area ---
st.title("🎓 Crescent University Chatbot")

for chat in st.session_state.history:
    st.chat_message(chat["role"]).write(chat["content"])

user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    if is_greeting(user_input):
        greeting_reply = random.choice([
            "Hello! How can I assist you today? 😊",
            "Hi there! Ask me anything about Crescent University.",
            "Welcome! I'm here to help with any university-related question."
        ])
        st.session_state.history.append({"role": "assistant", "content": greeting_reply})
    else:
        f_faculty = normalize_filter_value(selected_faculty)
        f_department = normalize_filter_value(selected_department)
        f_level = normalize_filter_value(selected_level)
        f_semester = normalize_filter_value(selected_semester)

        filtered_df = apply_filters(df, f_faculty, f_department, f_level, f_semester)

        if filtered_df.empty:
            st.session_state.history.append({"role": "assistant", "content": "⚠️ No matching data found for the selected filters."})
        else:
            cache_key = (f_faculty, f_department, f_level, f_semester)

            if cache_key not in st.session_state.embedding_cache:
                with st.spinner("Encoding and indexing..."):
                    embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
                    st.session_state.embedding_cache[cache_key] = embeddings
                    st.session_state.faiss_index_cache[cache_key] = build_faiss_index(embeddings)
            else:
                embeddings = st.session_state.embedding_cache[cache_key]

            index = st.session_state.faiss_index_cache[cache_key]

            query_embedding = model.encode([preprocess_text(user_input)], convert_to_numpy=True)
            D, I = index.search(query_embedding, 1)
            best_idx = I[0][0]

            context_qa = {
                "question": filtered_df.iloc[best_idx]["question"],
                "answer": filtered_df.iloc[best_idx]["answer"]
            }
            response = fallback_openai(user_input, context_qa)
            st.session_state.history.append({"role": "assistant", "content": response})
            st.session_state.related_questions = get_related_questions(user_input, filtered_df, index, model)
            
            # Rerun to immediately reflect new messages in UI
            st.experimental_rerun()

# --- Related Questions Section ---
if st.session_state.related_questions:
    st.subheader("🔍 Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q, key=f"rel_q_{q}"):
            st.session_state.history.append({"role": "user", "content": q})

            # Normalize filters again
            f_faculty = normalize_filter_value(selected_faculty)
            f_department = normalize_filter_value(selected_department)
            f_level = normalize_filter_value(selected_level)
            f_semester = normalize_filter_value(selected_semester)

            # Apply filters on dataframe
            filtered_df = apply_filters(df, f_faculty, f_department, f_level, f_semester)

            if filtered_df.empty:
                st.session_state.history.append({"role": "assistant", "content": "⚠️ No data available for these filters."})
            else:
                cache_key = (f_faculty, f_department, f_level, f_semester)

                # Retrieve embeddings and index or build if missing
                if cache_key not in st.session_state.embedding_cache or cache_key not in st.session_state.faiss_index_cache:
                    with st.spinner("Encoding and building index..."):
                        embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
                        st.session_state.embedding_cache[cache_key] = embeddings
                        index = build_faiss_index(embeddings)
                        st.session_state.faiss_index_cache[cache_key] = index
                else:
                    embeddings = st.session_state.embedding_cache[cache_key]
                    index = st.session_state.faiss_index_cache[cache_key]

                # Search best matching Q&A for the related question clicked
                with st.spinner("Searching for answer..."):
                    query_embedding = model.encode([preprocess_text(q)], convert_to_numpy=True)
                    D, I = index.search(query_embedding, 1)
                    best_idx = I[0][0]

                context_qa = {
                    "question": filtered_df.iloc[best_idx]["question"],
                    "answer": filtered_df.iloc[best_idx]["answer"]
                }
                with st.spinner("Generating response..."):
                    response = fallback_openai(q, context_qa)
                st.session_state.history.append({"role": "assistant", "content": response})

            # Rerun to update chat UI with new messages
            st.experimental_rerun()
