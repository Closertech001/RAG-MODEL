import streamlit as st
import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import re
import string
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Abbreviation Expansion ---
abbreviation_dict = {
    "cuab": "crescent university",
    "vc": "vice chancellor",
    "asuu": "academic staff union of universities",
    "phd": "doctor of philosophy",
    "bsc": "bachelor of science",
    "pgd": "postgraduate diploma",
    "msc": "master of science",
    "nysc": "national youth service corps"
}

def expand_abbreviations(text):
    words = text.split()
    expanded_words = [abbreviation_dict.get(word.lower(), word) for word in words]
    return " ".join(expanded_words)

# --- Spell Correction using TextBlob ---
def correct_spelling(text):
    return str(TextBlob(text).correct())

# --- Preprocessing ---
def preprocess(text):
    text = expand_abbreviations(text)
    text = correct_spelling(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# --- Embedding Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Load dataset and precompute embeddings ---
@st.cache_data
def load_data_and_embeddings():
    df = pd.read_csv("crescent_data.csv")
    df["question_clean"] = df["question"].apply(preprocess)
    df["embedding"] = df["question_clean"].apply(lambda q: model.encode(q, convert_to_tensor=True).cpu().numpy())
    return df

dataset = load_data_and_embeddings()

# --- Filter Function ---
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

# --- GPT Fallback ---
def gpt_fallback(prompt, model_name="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error contacting GPT model: {e}"

# --- Similar Question Finder ---
def find_similar_responses(user_question, data):
    user_question_processed = preprocess(user_question)
    user_embedding = model.encode(user_question_processed, convert_to_tensor=True).cpu().numpy()

    embeddings = np.stack(data["embedding"].values)
    similarities = cosine_similarity([user_embedding], embeddings)[0]

    top_indices = similarities.argsort()[::-1][:3]
    top_scores = similarities[top_indices]

    if top_scores[0] >= 0.75:
        responses = [(data.iloc[i]["question"], data.iloc[i]["answer"], data.iloc[i]["department"]) for i in top_indices]
        return responses, top_scores
    else:
        return [], top_scores

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.title("🎓 Crescent University Q&A Chatbot")

# Sidebar
with st.sidebar:
    st.title("Crescent University RAG Chatbot")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.subheader("📚 Filters")
    selected_faculty = st.selectbox("Faculty", ["All"] + sorted(dataset["faculty"].dropna().unique().tolist()))
    selected_department = st.selectbox("Department", ["All"] + sorted(dataset["department"].dropna().unique().tolist()))
    selected_level = st.selectbox("Level", ["All"] + sorted(dataset["level"].dropna().unique().tolist()))
    selected_semester = st.selectbox("Semester", ["All"] + sorted(dataset["semester"].dropna().unique().tolist()))

    st.subheader("⚙️ Model")
    model_choice = st.radio("Choose model", ["gpt-3.5-turbo", "gpt-4"], index=0)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
prompt = st.chat_input("Ask a question about Crescent University:")

if prompt:
    filtered_data = apply_filters(dataset, selected_faculty, selected_department, selected_level, selected_semester)

    if filtered_data.empty:
        answer = "No data matches the selected filters. Please adjust them and try again."
    else:
        responses, scores = find_similar_responses(prompt, filtered_data)

        if responses:
            main_answer = responses[0][1]
            related_qs = [r[0] for r in responses[1:]]
        else:
            main_answer = gpt_fallback(prompt, model_choice)
            related_qs = []

    st.session_state.chat_history.append((prompt, main_answer, related_qs))

# Display Chat
with st.container():
    for user_input, response, related in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {user_input}")
        st.markdown(f"**🤖 Bot:** {response}")
        if related:
            with st.expander("Related Questions"):
                for q in related:
                    if st.button(q, key=q):
                        sub_responses, _ = find_similar_responses(q, apply_filters(dataset, selected_faculty, selected_department, selected_level, selected_semester))
                        if sub_responses:
                            answer = sub_responses[0][1]
                        else:
                            answer = gpt_fallback(q, model_choice)
                        st.session_state.chat_history.append((q, answer, []))
                        st.rerun()
        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OpenAI, and SentenceTransformers")
