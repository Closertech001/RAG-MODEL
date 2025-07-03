import json
import os
import re
import faiss
import numpy as np
import openai
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")

# Hardcoded abbreviations and synonyms
ABBREVIATIONS = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "cv": "computer vision",
    "ds": "data science",
    "cs": "computer science",
    "u": "you",
    "r": "are",
    "ur": "your"
}

SYNONYMS = {
    "study": "learn",
    "teach": "educate",
    "trainer": "instructor",
    "professor": "instructor",
    "school": "university",
    "uni": "university",
    "lecture": "class",
    "learnt": "learned"
}

DEFAULT_VOCAB = [
    "learn", "study", "education", "instructor", "professor", "university",
    "student", "machine learning", "artificial intelligence", "class",
    "overfitting", "underfitting", "data", "model", "training", "exam"
]

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Create a 'content' field by merging 'question' and 'answer'
    for item in data:
        item["content"] = item["question"] + " " + item["answer"]
        
    return [item["content"] for item in data], data

# Build FAISS index
def build_index(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, model, embeddings

# Semantic search
def search(query, index, model, chunks, top_k=1):
    query_vec = model.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k=top_k)
    return [chunks[i] for i in I[0]], float(D[0][0])

# Spelling correction
def correct_spelling(text):
    return str(TextBlob(text).correct())

# Semantic normalization of each word
def semantic_normalize(input_text, vocab_list, model):
    input_tokens = input_text.lower().split()
    normalized_tokens = []
    for token in input_tokens:
        token_emb = model.encode(token, convert_to_tensor=True)
        vocab_embs = model.encode(vocab_list, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(token_emb, vocab_embs)[0]
        best_match_idx = scores.argmax().item()
        best_match = vocab_list[best_match_idx]
        normalized_tokens.append(best_match)
    return " ".join(normalized_tokens)

# Full input normalization pipeline
def normalize_input(text, vocab_list, model):
    corrected = correct_spelling(text)
    text = corrected.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    processed = []
    for word in words:
        if word in ABBREVIATIONS:
            processed.append(ABBREVIATIONS[word])
        elif word in SYNONYMS:
            processed.append(SYNONYMS[word])
        else:
            processed.append(word)
    text = " ".join(processed)
    return semantic_normalize(text, DEFAULT_VOCAB, model)

# GPT generation with memory + limit
def ask_gpt_with_memory(messages, max_history=6, model="gpt-3.5-turbo"):
    if len(messages) > max_history + 1:
        messages = [messages[0]] + messages[-max_history:]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message["content"].strip(), response["usage"]
    except Exception as e:
        return f"Error: {e}", {"prompt_tokens": 0, "completion_tokens": 0}

# Feedback logging
def log_feedback(query, answer, rating):
    with open("feedback.csv", "a") as f:
        f.write(f"{datetime.now()},\"{query}\",\"{answer}\",{rating}\n")

# Small talk detection
def is_small_talk(query):
    patterns = ["hello", "hi", "how are you", "your name", "tell me a joke"]
    return any(p in query.lower() for p in patterns)

def handle_small_talk(query):
    if "hello" in query.lower() or "hi" in query.lower():
        return "Hi there! How can I assist you today?"
    if "joke" in query.lower():
        return "Why don't scientists trust atoms? Because they make up everything!"
    if "your name" in query.lower():
        return "I’m CrescentBot, your university assistant."
    return "I'm here to help with anything academic-related!"
