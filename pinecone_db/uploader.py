"""
Handles upserting, updating, and deleting vectors in Pinecone.
Supports batching and async-style chunked uploads for speed.
"""

from typing import List

from config import settings
from models.document import Chunk
from pinecone_db.pinecone_manager import PineconeManager
from embeddings.embedder import Embedder
from utils.helper import retry
from utils.logger import get_logger

logger = get_logger("uploader")

UPSERT_BATCH_SIZE = 100


class Uploader:
    def __init__(self, manager: PineconeManager, embedder: Embedder):
        self.manager = manager
        self.embedder = embedder

    @retry(max_attempts=3)
    def _upsert_batch(self, vectors: list, namespace: str):
        self.manager.index.upsert(vectors=vectors, namespace=namespace)

    def upload_chunks(self, chunks: List[Chunk], namespace: str = None) -> int:
        if not chunks:
            logger.warning("No chunks to upload.")
            return 0

        namespace = namespace or settings.retrieval.default_namespace
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        vectors_emb = self.embedder.embed_batch(texts)

        vectors = [
            {"id": chunk.id, "values": emb, "metadata": chunk.to_metadata()}
            for chunk, emb in zip(chunks, vectors_emb)
        ]

        total_uploaded = 0
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            self._upsert_batch(batch, namespace)
            total_uploaded += len(batch)
            logger.info(f"Uploaded batch {i // UPSERT_BATCH_SIZE + 1}: {len(batch)} vectors.")

        logger.info(f"Total vectors uploaded to namespace '{namespace}': {total_uploaded}")
        return total_uploaded

    def update_vector(self, vector_id: str, new_text: str, namespace: str = None, extra_metadata: dict = None):
        namespace = namespace or settings.retrieval.default_namespace
        embedding = self.embedder.embed_text(new_text)
        metadata = {"text": new_text}
        if extra_metadata:
            metadata.update(extra_metadata)
        self.manager.index.update(id=vector_id, values=embedding, set_metadata=metadata, namespace=namespace)
        logger.info(f"Updated vector '{vector_id}' in namespace '{namespace}'.")

    def delete_by_id(self, vector_id: str, namespace: str = None):
        namespace = namespace or settings.retrieval.default_namespace
        self.manager.index.delete(ids=[vector_id], namespace=namespace)
        logger.info(f"Deleted vector '{vector_id}' from namespace '{namespace}'.")

    def delete_by_source(self, source_filename: str, namespace: str = None):
        namespace = namespace or settings.retrieval.default_namespace
        self.manager.index.delete(filter={"source": {"$eq": source_filename}}, namespace=namespace)
        logger.info(f"Deleted all vectors for source '{source_filename}' in namespace '{namespace}'.")

    def delete_namespace(self, namespace: str):
        self.manager.index.delete(delete_all=True, namespace=namespace)
        logger.warning(f"Deleted entire namespace '{namespace}'.")
