"""
Provider-independent LLM client. Switch providers purely via config/.env —
no code changes needed. Supports: Gemini, OpenAI, Groq, Ollama (offline).
"""

from typing import Generator

from config import settings
from utils.helper import retry
from utils.logger import get_logger

logger = get_logger("llm_client")


class LLMClient:
    def __init__(self, provider: str = None):
        self.provider = (provider or settings.llm.provider).lower()
        self._client = None
        self._init_provider()

    def _init_provider(self):
        if self.provider == "gemini":
            import google.generativeai as genai

            if not settings.llm.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set.")
            genai.configure(api_key=settings.llm.gemini_api_key)
            self._client = genai.GenerativeModel(settings.llm.gemini_model)

        elif self.provider == "openai":
            from openai import OpenAI

            if not settings.llm.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = OpenAI(api_key=settings.llm.openai_api_key)

        elif self.provider == "groq":
            from groq import Groq

            if not settings.llm.groq_api_key:
                raise ValueError("GROQ_API_KEY not set.")
            self._client = Groq(api_key=settings.llm.groq_api_key)

        elif self.provider == "ollama":
            import ollama

            self._client = ollama.Client(host=settings.llm.ollama_base_url)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info(f"LLM provider initialized: {self.provider}")

    @retry(max_attempts=3, delay_seconds=2.0)
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self.provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self._client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": settings.llm.temperature,
                    "max_output_tokens": settings.llm.max_tokens,
                },
            )
            return response.text

        elif self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat.completions.create(
                model=settings.llm.openai_model,
                messages=messages,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "groq":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat.completions.create(
                model=settings.llm.groq_model,
                messages=messages,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "ollama":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat(
                model=settings.llm.ollama_model,
                messages=messages,
                options={"temperature": settings.llm.temperature},
            )
            return response["message"]["content"]

    def generate_stream(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """Yields tokens/chunks as they arrive, for streaming UIs."""
        if self.provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            stream = self._client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": settings.llm.temperature,
                    "max_output_tokens": settings.llm.max_tokens,
                },
                stream=True,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        elif self.provider in ("openai", "groq"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            model = settings.llm.openai_model if self.provider == "openai" else settings.llm.groq_model
            stream = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        elif self.provider == "ollama":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            stream = self._client.chat(
                model=settings.llm.ollama_model,
                messages=messages,
                options={"temperature": settings.llm.temperature},
                stream=True,
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
