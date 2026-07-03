"""
Wraps SentenceTransformer for configurable, batched embedding generation.
"""

from typing import List

from config import settings
from utils.logger import get_logger

logger = get_logger("embedder")

_MODEL_CACHE = {}


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding.model_name
        self.model = self._load_model(self.model_name)

    def _load_model(self, model_name: str):
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        return model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        batch_size = batch_size or settings.embedding.batch_size
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return [v.tolist() for v in vectors]
