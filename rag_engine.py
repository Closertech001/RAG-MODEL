# rag_engine.py
import os
import json
import re
import numpy as np
import openai
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import faiss
from config import ABBREVIATIONS, SYNONYMS, DEFAULT_VOCAB
from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["content"] = item["question"] + " " + item["answer"]
    return [item["content"] for item in data], data

def build_index(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, model, embeddings

def search(query, index, model, chunks, top_k=1):
    query_vec = model.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k=top_k)
    return [chunks[i] for i in I[0]], float(D[0][0])

def correct_spelling(text):
    return str(TextBlob(text).correct())

def semantic_normalize(input_text, vocab_list, model):
    tokens = input_text.lower().split()
    normalized = []
    for token in tokens:
        token_emb = model.encode(token, convert_to_tensor=True)
        vocab_embs = model.encode(vocab_list, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(token_emb, vocab_embs)[0]
        best_match = vocab_list[scores.argmax().item()]
        normalized.append(best_match)
    return " ".join(normalized)

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
    return semantic_normalize(text, vocab_list, model)

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
