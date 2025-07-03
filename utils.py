# utils.py

import json
import os

def convert_file_to_chunks(file, faculty, department, level="100"):
    ext = file.name.split('.')[-1]
    try:
        if ext == "txt":
            text = file.read().decode("utf-8")
        elif ext == "csv":
            text = file.getvalue().decode("utf-8")
        elif ext in ["pdf", "docx"]:
            text = f"{ext.upper()} file uploaded. Parsing not yet implemented."
        else:
            text = "Unsupported file type."
    except:
        text = "Unable to read content from file."

    return [{
        "question": f"What is the content of the uploaded {ext.upper()} file?",
        "answer": text,
        "faculty": faculty,
        "department": department,
        "level": level
    }]

def append_chunks_to_json(new_data, file_path="qa_dataset.json"):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(new_data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)
