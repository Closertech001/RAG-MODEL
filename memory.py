# memory.py
from collections import deque

class ContextMemory:
    def __init__(self, max_turns=6):
        self.history = deque(maxlen=max_turns)

    def add_turn(self, user, bot):
        self.history.append({"user": user, "bot": bot})

    def get_context(self):
        return self.history
