"""
Semantic search over Pinecone with score thresholding, metadata filtering,
and duplicate removal.
"""

from typing import List, Optional

from config import settings
from models.document import RetrievedChunk
from pinecone_db.pinecone_manager import PineconeManager
from embeddings.embedder import Embedder
from utils.logger import get_logger

logger = get_logger("search")


class SemanticSearch:
    def __init__(self, manager: PineconeManager, embedder: Embedder):
        self.manager = manager
        self.embedder = embedder

    def search(
        self,
        query: str,
        namespace: str = None,
        top_k: int = None,
        score_threshold: float = None,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        namespace = namespace or settings.retrieval.default_namespace
        top_k = top_k or settings.retrieval.top_k
        score_threshold = (
            score_threshold if score_threshold is not None else settings.retrieval.similarity_threshold
        )

        query_vector = self.embedder.embed_text(query)

        results = self.manager.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=metadata_filter,
        )

        retrieved = []
        seen_texts = set()

        for match in results.get("matches", []):
            score = match["score"]
            if score < score_threshold:
                continue

            metadata = match.get("metadata", {})
            text = metadata.get("text", "")

            # duplicate removal by exact text match
            dedup_key = text.strip()[:300]
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            retrieved.append(
                RetrievedChunk(
                    id=match["id"],
                    text=text,
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page"),
                    score=score,
                    metadata=metadata,
                )
            )

        # rank by score descending (Pinecone already does this, but re-affirm post-filter)
        retrieved.sort(key=lambda r: r.score, reverse=True)

        logger.info(f"Query '{query[:60]}...' -> {len(retrieved)} chunks above threshold {score_threshold}.")
        return retrieved
