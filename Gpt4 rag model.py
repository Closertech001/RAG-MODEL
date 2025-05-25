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
from openai import OpenAI

# --- Check API key ---
if not os.getenv("OPENAI_API_KEY"):
    st.error("\u274c OPENAI_API_KEY not set in environment.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- SymSpell setup ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- Abbreviations dictionary ---
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
        "hi", "hello", "hey", "hi there", "greetings", "how are you", "how are you doing",
        "how's it going", "can we talk?", "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    return text.lower().strip() in greetings

@st.cache_data
def load_data():
    try:
        with open("qa_dataset.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        st.error(f"\u274c Failed to load qa_dataset.json: {e}")
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

@st.cache_resource
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

    for model_name in ["gpt-4o", "gpt-3.5-turbo"]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3
            )
            st.info(f"\u2705 Response from: {model_name}")
            return response.choices[0].message.content.strip()
        except Exception:
            st.warning(f"\u26a0\ufe0f {model_name} failed. Trying fallback model...")
            continue

    return "\u274c Sorry, I'm currently unable to answer. Please try again later."

def get_related_questions(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)
    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [df.iloc[i]['question'] for i in I[0] if i < len(df)]

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent University RAG Chatbot", page_icon="\ud83c\udf93", layout="wide")

df = load_data()
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

# Sidebar Filters
with st.sidebar:
    st.title("Crescent University RAG Chatbot")
    if st.button("\ud83d\uddd1\ufe0f Clear Chat"):
        st.session_state.history = []
        st.session_state.related_questions = []
        st.experimental_rerun()

    st.subheader("\ud83d\udcda Filters")
    selected_faculty = st.selectbox("Faculty", ["All"] + sorted(df["faculty"].dropna().unique().tolist()))
    selected_department = st.selectbox("Department", ["All"] + sorted(df["department"].dropna().unique().tolist()))
    selected_level = st.selectbox("Level", ["All"] + sorted(df["level"].dropna().unique().tolist()))
    selected_semester = st.selectbox("Semester", ["All"] + sorted(df["semester"].dropna().unique().tolist()))

# Display chat history
st.title("\ud83c\udf93 Crescent University Chatbot")
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        st.chat_message("assistant").write(chat["content"])

# Chat input
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)

    if not filtered_df.empty:
        cache_key = (selected_faculty, selected_department, selected_level, selected_semester)
        if cache_key in st.session_state.embedding_cache:
            embeddings = st.session_state.embedding_cache[cache_key]
        else:
            embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
            st.session_state.embedding_cache[cache_key] = embeddings

        index = build_faiss_index(embeddings)

        clean_input = preprocess_text(user_input)
        top_k = 1
        query_embedding = model.encode([clean_input], convert_to_numpy=True)
        D, I = index.search(query_embedding, top_k)
        best_idx = I[0][0] if I[0][0] < len(filtered_df) else None
        context_qa = filtered_df.iloc[best_idx] if best_idx is not None else None
        response = fallback_openai(user_input, context_qa)

        st.session_state.history.append({"role": "assistant", "content": response})
        st.session_state.related_questions = get_related_questions(user_input, filtered_df, index, model)
    else:
        st.session_state.history.append({
            "role": "assistant",
            "content": "\u26a0\ufe0f No matching data found for the selected filters."
        })

# Related questions buttons
if st.session_state.related_questions:
    st.subheader("\ud83d\udd0d Related Questions:")
    for q in st.session_state.related_questions:
        if st.button(q):
            st.session_state.history.append({"role": "user", "content": q})
            filtered_df = apply_filters(df, selected_faculty, selected_department, selected_level, selected_semester)
            if not filtered_df.empty:
                cache_key = (selected_faculty, selected_department, selected_level, selected_semester)
                embeddings = st.session_state.embedding_cache.get(cache_key)
                if embeddings is None:
                    embeddings = model.encode(filtered_df["text"].tolist(), convert_to_numpy=True)
                    st.session_state.embedding_cache[cache_key] = embeddings
                index = build_faiss_index(embeddings)
                clean_input = preprocess_text(q)
                query_embedding = model.encode([clean_input], convert_to_numpy=True)
                D, I = index.search(query_embedding, 1)
                best_idx = I[0][0] if I[0][0] < len(filtered_df) else None
                context_qa = filtered_df.iloc[best_idx] if best_idx is not None else None
                response = fallback_openai(q, context_qa)
                st.session_state.history.append({"role": "assistant", "content": response})
