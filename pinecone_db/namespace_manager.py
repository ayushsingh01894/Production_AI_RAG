"""
Utility helpers for managing Pinecone namespaces (multi-tenant isolation).
"""

from pinecone_db.pinecone_manager import PineconeManager
from utils.logger import get_logger

logger = get_logger("namespace_manager")


class NamespaceManager:
    def __init__(self, manager: PineconeManager):
        self.manager = manager

    def list_namespaces(self) -> dict:
        """Returns {namespace: vector_count} from index stats."""
        stats = self.manager.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return {ns: info.get("vector_count", 0) for ns, info in namespaces.items()}

    def namespace_exists(self, namespace: str) -> bool:
        return namespace in self.list_namespaces()

    def vector_count(self, namespace: str) -> int:
        return self.list_namespaces().get(namespace, 0)

    def delete_namespace(self, namespace: str):
        self.manager.index.delete(delete_all=True, namespace=namespace)
        logger.warning(f"Namespace '{namespace}' fully deleted.")
