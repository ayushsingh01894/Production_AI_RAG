"""
In-session conversation memory with a configurable rolling history window.
"""

from typing import List

from config import settings


class ChatHistory:
    def __init__(self, window: int = None):
        self.window = window or settings.memory.history_window
        self._history: List[dict] = []

    def add_turn(self, question: str, answer: str):
        self._history.append({"question": question, "answer": answer})
        # keep only the last N turns
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]

    def get_history(self) -> List[dict]:
        return list(self._history)

    def clear(self):
        self._history = []

    def __len__(self):
        return len(self._history)
