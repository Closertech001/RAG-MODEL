# feedback.py
import csv
import os
from datetime import datetime

FEEDBACK_FILE = "feedback.csv"

def log_feedback(query, response, rating):
    with open(FEEDBACK_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), query, response, rating])

def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, newline="") as file:
        reader = csv.reader(file)
        return list(reader)

def analyze_feedback():
    feedback = load_feedback()
    good, bad = 0, 0
    for row in feedback:
        if len(row) >= 4:
            if row[3] == "👍":
                good += 1
            elif row[3] == "👎":
                bad += 1
    return good, bad
