"""
Builds the final prompt sent to the LLM: system prompt + retrieved context +
conversation history + user question. Also supports different prompt "modes"
(Teacher, Interviewer, Technical Expert, Beginner, Summarizer).
"""

import os
from typing import List

from models.document import RetrievedChunk

SYSTEM_PROMPT_PATH = os.path.join("prompts", "system_prompt.txt")

PROMPT_MODES = {
    "default": "",
    "teacher": (
        "Explain the answer the way a patient teacher would to a student — "
        "use simple language, analogies, and check understanding at the end."
    ),
    "interviewer": (
        "Answer as if you are being interviewed about this topic — confident, "
        "structured, and to the point, as you would in a technical interview."
    ),
    "technical_expert": (
        "Answer with full technical depth and precision, using correct terminology, "
        "as an expert speaking to another expert."
    ),
    "beginner": (
        "Answer assuming the reader has no prior background — avoid jargon, "
        "define any technical term you use."
    ),
    "summarizer": (
        "Answer in a compact summary — 3-5 bullet points capturing only the key facts."
    ),
}


class PromptBuilder:
    def __init__(self):
        self.base_system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        if os.path.exists(SYSTEM_PROMPT_PATH):
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are a helpful assistant that answers based on provided context."

    def build_context_block(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context was found in the documents."

        blocks = []
        for i, c in enumerate(chunks, start=1):
            page_info = f", page {c.page}" if c.page else ""
            blocks.append(f"[Source {i}: {c.source}{page_info} | similarity={c.score:.2f}]\n{c.text}")
        return "\n\n---\n\n".join(blocks)

    def build_history_block(self, history: List[dict]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def build_prompt(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        history: List[dict] = None,
        mode: str = "default",
    ) -> tuple:
        """Returns (system_prompt, user_prompt)."""
        mode_instruction = PROMPT_MODES.get(mode, "")
        system_prompt = self.base_system_prompt
        if mode_instruction:
            system_prompt += f"\n\nStyle instruction: {mode_instruction}"

        context_block = self.build_context_block(chunks)
        history_block = self.build_history_block(history or [])

        parts = []
        if history_block:
            parts.append(f"Conversation so far:\n{history_block}")
        parts.append(f"Context from documents:\n{context_block}")
        parts.append(f"Question: {question}")

        user_prompt = "\n\n".join(parts)
        return system_prompt, user_prompt
