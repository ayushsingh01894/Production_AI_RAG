"""
Generates the final AI answer, wiring together the LLM client and prompt
builder, with an optional response cache to instantly answer repeated
questions.
"""

import os
from typing import Generator, List

from config import settings
from llm.llm_client import LLMClient
from llm.prompt_builder import PromptBuilder
from models.document import RetrievedChunk
from utils.helper import compute_text_hash, load_json, save_json
from utils.logger import get_logger

logger = get_logger("response_generator")

CACHE_FILE = os.path.join(settings.misc.cache_dir, "response_cache.json")


class ResponseGenerator:
    def __init__(self, llm_client: LLMClient = None, prompt_builder: PromptBuilder = None):
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.cache = load_json(CACHE_FILE, default={}) if settings.misc.cache_enabled else {}

    def _cache_key(self, question: str, namespace: str, mode: str) -> str:
        return compute_text_hash(f"{namespace}:{mode}:{question.strip().lower()}")

    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        history: List[dict] = None,
        namespace: str = "default",
        mode: str = "default",
        use_cache: bool = True,
    ) -> tuple:
        """Returns (answer_text, from_cache: bool)."""
        key = self._cache_key(question, namespace, mode)

        if use_cache and settings.misc.cache_enabled and key in self.cache:
            logger.info("Cache hit — returning instant response.")
            return self.cache[key], True

        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            question, chunks, history, mode
        )
        answer = self.llm_client.generate(user_prompt, system_prompt)

        if settings.misc.cache_enabled:
            self.cache[key] = answer
            save_json(CACHE_FILE, self.cache)

        return answer, False

    def generate_answer_stream(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        history: List[dict] = None,
        mode: str = "default",
    ) -> Generator[str, None, None]:
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            question, chunks, history, mode
        )
        yield from self.llm_client.generate_stream(user_prompt, system_prompt)
