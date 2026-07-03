"""
Smart recursive character-based chunker with overlap and metadata preservation.
"""

import os
from typing import List, Tuple

from config import settings
from models.document import Chunk
from utils.helper import compute_text_hash, generate_id
from utils.logger import get_logger

logger = get_logger("chunker")

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """
    Splits text recursively on a hierarchy of separators (paragraph -> line ->
    sentence -> word -> char) so chunks break on natural boundaries whenever
    possible, while respecting chunk_size and chunk_overlap.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # base case: hard split by characters
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        parts = text.split(sep)
        chunks, current = [], ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_text(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._merge_with_overlap(chunks)

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            merged.append((overlap_text + " " + chunk).strip())
        return merged

    def chunk_pages(
        self, pages: List[Tuple[int, str]], source: str, namespace: str = "default"
    ) -> List[Chunk]:
        """pages: list of (page_number, text) -> list[Chunk]"""
        doc_hash = compute_text_hash(source + str(len(pages)))
        chunks: List[Chunk] = []
        chunk_counter = 0

        for page_num, page_text in pages:
            pieces = self._split_text(page_text.strip(), DEFAULT_SEPARATORS)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    Chunk(
                        id=generate_id("chunk"),
                        text=piece,
                        source=os.path.basename(source),
                        page=page_num,
                        chunk_index=chunk_counter,
                        doc_hash=doc_hash,
                        namespace=namespace,
                    )
                )
                chunk_counter += 1

        logger.info(f"Chunked '{os.path.basename(source)}' into {len(chunks)} chunks.")
        return chunks
