"""
Central configuration for the Production AI RAG Assistant.
All tunable settings live here, loaded from environment variables (.env).
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


@dataclass
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "production-rag-index")
    cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    region: str = os.getenv("PINECONE_REGION", "us-east-1")
    metric: str = "cosine"


@dataclass
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    dimension: int = _get_int("EMBEDDING_DIMENSION", 384)
    batch_size: int = 64


@dataclass
class ChunkingConfig:
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 120)


@dataclass
class RetrievalConfig:
    top_k: int = _get_int("TOP_K", 5)
    similarity_threshold: float = _get_float("SIMILARITY_THRESHOLD", 0.35)
    default_namespace: str = os.getenv("DEFAULT_NAMESPACE", "default")


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai | groq | ollama
    temperature: float = _get_float("TEMPERATURE", 0.3)
    max_tokens: int = _get_int("MAX_TOKENS", 1024)

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")


@dataclass
class MemoryConfig:
    history_window: int = _get_int("CHAT_HISTORY_WINDOW", 6)


@dataclass
class MiscConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    cache_enabled: bool = _get_bool("CACHE_ENABLED", True)
    log_dir: str = "logs"
    cache_dir: str = "cache"
    data_dir: str = "data"


@dataclass
class Settings:
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)


settings = Settings()
