"""
Thin orchestration layer over SemanticSearch — the "retriever" in RAG.
"""

from typing import List, Optional

from models.document import RetrievedChunk
from pinecone_db.search import SemanticSearch


class Retriever:
    def __init__(self, search: SemanticSearch):
        self.search = search

    def retrieve(
        self,
        query: str,
        namespace: str = None,
        top_k: int = None,
        score_threshold: float = None,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        return self.search.search(
            query=query,
            namespace=namespace,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )
