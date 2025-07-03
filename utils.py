# --- utils.py ---
import os
import fitz  # PyMuPDF
import docx
import pandas as pd
import json
from uuid import uuid4

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# Extract text from TXT/CSV
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Convert uploaded file to JSON chunks
def convert_file_to_chunks(file, faculty, department, level):
    if file.name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        text = extract_text_from_docx(file)
    elif file.name.endswith(".csv") or file.name.endswith(".txt"):
        text = extract_text_from_txt(file)
    else:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    return [
        {
            "id": str(uuid4()),
            "content": para,
            "faculty": faculty,
            "department": department,
            "level": level
        } for para in paragraphs
    ]

# Save uploaded chunks to JSON
def append_chunks_to_json(new_chunks, json_path="data/data.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.extend(new_chunks)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
