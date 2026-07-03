"""
Manages the Pinecone connection and index lifecycle (create/connect/delete).
"""

from config import settings
from utils.helper import retry
from utils.logger import get_logger

logger = get_logger("pinecone_manager")


class PineconeManager:
    def __init__(self):
        from pinecone import Pinecone

        if not settings.pinecone.api_key:
            raise ValueError("PINECONE_API_KEY is not set. Please configure your .env file.")

        self.pc = Pinecone(api_key=settings.pinecone.api_key)
        self.index_name = settings.pinecone.index_name
        self._index = None

    @retry(max_attempts=3)
    def create_index_if_not_exists(self, dimension: int = None):
        from pinecone import ServerlessSpec

        dimension = dimension or settings.embedding.dimension
        existing = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name in existing:
            logger.info(f"Index '{self.index_name}' already exists. Connecting.")
        else:
            logger.info(f"Creating index '{self.index_name}' (dim={dimension})...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=settings.pinecone.metric,
                spec=ServerlessSpec(
                    cloud=settings.pinecone.cloud, region=settings.pinecone.region
                ),
            )
            logger.info(f"Index '{self.index_name}' created successfully.")

        self._index = self.pc.Index(self.index_name)
        return self._index

    @property
    def index(self):
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def delete_index(self):
        logger.warning(f"Deleting index '{self.index_name}'...")
        self.pc.delete_index(self.index_name)
        self._index = None

    def describe_index_stats(self) -> dict:
        return self.index.describe_index_stats()

    def list_indexes(self) -> list:
        return [idx.name for idx in self.pc.list_indexes()]
