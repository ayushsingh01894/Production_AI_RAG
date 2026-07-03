"""
The central RAG pipeline — wires together loading, chunking, embedding,
Pinecone upload/search, LLM generation, and memory. Both the CLI (app.py)
and the Streamlit UI (streamlit_app.py) call into this single pipeline so
there is zero logic duplication between frontends.
"""

import os
import time
from typing import List, Optional

from config import settings
from chunking.chunker import RecursiveChunker
from embeddings.embedder import Embedder
from loaders.pdf_loader import DocumentLoader
from llm.llm_client import LLMClient
from llm.prompt_builder import PromptBuilder
from llm.response_generator import ResponseGenerator
from memory.chat_history import ChatHistory
from models.document import RAGAnswer
from pinecone_db.namespace_manager import NamespaceManager
from pinecone_db.pinecone_manager import PineconeManager
from pinecone_db.search import SemanticSearch
from pinecone_db.uploader import Uploader
from rag.retrieval import Retriever
from utils.helper import compute_file_hash
from utils.logger import get_logger

logger = get_logger("pipeline")


class RAGPipeline:
    def __init__(self, llm_provider: str = None):
        self.embedder = Embedder()
        self.manager = PineconeManager()
        self.manager.create_index_if_not_exists(dimension=self.embedder.dimension)

        self.loader = DocumentLoader()
        self.chunker = RecursiveChunker()
        self.uploader = Uploader(self.manager, self.embedder)
        self.search = SemanticSearch(self.manager, self.embedder)
        self.retriever = Retriever(self.search)
        self.namespace_manager = NamespaceManager(self.manager)

        self.llm_client = LLMClient(provider=llm_provider)
        self.prompt_builder = PromptBuilder()
        self.response_generator = ResponseGenerator(self.llm_client, self.prompt_builder)

        self.chat_history = ChatHistory()
        self._uploaded_hashes = set()

    # ---------- Document ingestion ----------

    def upload_pdf(self, file_path: str, namespace: str = None) -> int:
        namespace = namespace or settings.retrieval.default_namespace
        file_hash = compute_file_hash(file_path)

        if file_hash in self._uploaded_hashes:
            logger.info(f"Skipping duplicate file: {os.path.basename(file_path)}")
            return 0

        pages = self.loader.load(file_path)
        chunks = self.chunker.chunk_pages(pages, source=file_path, namespace=namespace)
        count = self.uploader.upload_chunks(chunks, namespace=namespace)
        self._uploaded_hashes.add(file_hash)
        return count

    def upload_folder(self, folder_path: str, namespace: str = None) -> dict:
        files = self.loader.load_folder(folder_path)
        results = {}
        for file_path in files:
            try:
                results[os.path.basename(file_path)] = self.upload_pdf(file_path, namespace)
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")
                results[os.path.basename(file_path)] = f"ERROR: {e}"
        return results

    # ---------- Search / Ask ----------

    def semantic_search(
        self, query: str, namespace: str = None, top_k: int = None, metadata_filter: Optional[dict] = None
    ):
        return self.retriever.retrieve(
            query=query, namespace=namespace, top_k=top_k, metadata_filter=metadata_filter
        )

    def ask(
        self,
        question: str,
        namespace: str = None,
        top_k: int = None,
        mode: str = "default",
        use_history: bool = True,
        use_cache: bool = True,
    ) -> RAGAnswer:
        start = time.time()
        namespace = namespace or settings.retrieval.default_namespace

        chunks = self.retriever.retrieve(question, namespace=namespace, top_k=top_k)
        history = self.chat_history.get_history() if use_history else []

        answer_text, from_cache = self.response_generator.generate_answer(
            question=question,
            chunks=chunks,
            history=history,
            namespace=namespace,
            mode=mode,
            use_cache=use_cache,
        )

        if use_history:
            self.chat_history.add_turn(question, answer_text)

        latency = time.time() - start

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            namespace=namespace,
            latency_seconds=latency,
            from_cache=from_cache,
        )

    def ask_stream(self, question: str, namespace: str = None, top_k: int = None, mode: str = "default"):
        """Generator version for Streamlit's write_stream."""
        namespace = namespace or settings.retrieval.default_namespace
        chunks = self.retriever.retrieve(question, namespace=namespace, top_k=top_k)
        history = self.chat_history.get_history()

        full_answer = ""
        for token in self.response_generator.generate_answer_stream(question, chunks, history, mode):
            full_answer += token
            yield token, chunks

        self.chat_history.add_turn(question, full_answer)

    # ---------- Management ----------

    def get_statistics(self) -> dict:
        stats = self.manager.describe_index_stats()
        namespaces = self.namespace_manager.list_namespaces()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", self.embedder.dimension),
            "namespaces": namespaces,
            "embedding_model": self.embedder.model_name,
            "llm_provider": self.llm_client.provider,
        }

    def delete_document(self, source_filename: str, namespace: str = None):
        namespace = namespace or settings.retrieval.default_namespace
        self.uploader.delete_by_source(source_filename, namespace=namespace)

    def delete_namespace(self, namespace: str):
        self.namespace_manager.delete_namespace(namespace)

    def clear_chat_history(self):
        self.chat_history.clear()
