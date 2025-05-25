import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

# Step 0: Setup and spell correction
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you", "how are you doing",
                 "how's it going", "can we talk?", "can we have a conversation?", "okay", "i'm fine", "i am fine"]
    return text.lower().strip() in greetings

# Step 1: Load raw dataset
with open("qa_dataset (1).json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Step 2: Prepare combined Q&A entries for RAG
rag_data = []
for entry in raw_data:
    question = entry.get("question", "").strip()
    answer = entry.get("answer", "").strip()
    if question and answer:
        combined_text = f"Q: {question}\nA: {answer}"
        rag_data.append({"text": combined_text, "question": question})

# Step 3: Convert to DataFrame
df = pd.DataFrame(rag_data)

# Step 4: Load sentence transformer model for embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

# Step 5: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 6: Save embeddings and data
faiss.write_index(index, "rag_index.faiss")
df.to_csv("rag_ready_dataset.csv", index=False, encoding="utf-8")

print("✅ FAISS index saved as 'rag_index.faiss'")
print("✅ Dataset saved as 'rag_ready_dataset.csv'")

# Step 7: Define GPT-4 RAG function
def query_gpt4_with_context(user_query, df, index, model, top_k=5):
    clean_query = preprocess_text(user_query)

    if is_greeting(clean_query):
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!", "I'm doing well, thank you!", "Sure pal", "Okay", "I'm fine, thank you"])

    query_embedding = model.encode([clean_query])
    D, I = index.search(np.array(query_embedding), top_k)

    context_blocks = []
    for i in I[0]:
        if i < len(df):
            context_blocks.append(df.iloc[i]['text'])

    context_string = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant for Crescent University. "
        "Use the provided information to answer the user's question accurately. "
        "If the answer is not clearly stated, say you don't know and suggest referring to official sources."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Based on the following context:\n\n{context_string}\n\nAnswer the following question:\n{user_query}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"❌ Error calling GPT-4: {str(e)}"

# Example usage (uncomment to test)
# user_question = "What is the duration of MTH 101?"
# print(query_gpt4_with_context(user_question, df, index, model))
