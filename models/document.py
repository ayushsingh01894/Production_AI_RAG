"""
Core data models used across the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""

    id: str
    text: str
    source: str                 # file name
    page: Optional[int] = None
    chunk_index: int = 0
    doc_hash: str = ""
    namespace: str = "default"
    extra_metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        meta = {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "doc_hash": self.doc_hash,
        }
        if self.page is not None:
            meta["page"] = self.page
        meta.update(self.extra_metadata)
        return meta


@dataclass
class RetrievedChunk:
    """A chunk returned from Pinecone with its similarity score."""

    id: str
    text: str
    source: str
    page: Optional[int]
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGAnswer:
    """Final answer returned to the user."""

    question: str
    answer: str
    sources: list  # list[RetrievedChunk]
    namespace: str
    latency_seconds: float = 0.0
    from_cache: bool = False
