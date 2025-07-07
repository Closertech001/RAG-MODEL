# config.py

BOT_NAME = "CrescentBot"

BOT_PERSONALITY = {
    "name": BOT_NAME,
    "traits": {
        "friendly": True,
        "empathetic": True,
        "helpful": True,
    },
    "greeting": "Hi there! I'm CrescentBot, your academic assistant 😊",
    "fallback": "I'm not sure about that yet, but I’ll try to find out!",
    "clarify": "Could you please clarify your question a bit more?",
}

SMALL_TALK_PATTERNS = [
    "hello", "hi", "who are you", "your name", "thanks", "thank you",
    "tell me a joke", "how are you", "good morning", "good afternoon"
]

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
