"""
Loads documents (PDF, TXT, DOCX, Markdown) and extracts raw page-wise text.
"""

import os
from typing import List, Tuple

from pypdf import PdfReader

from utils.logger import get_logger

logger = get_logger("loader")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class DocumentLoader:
    """Loads a single file and returns [(page_number, text), ...]."""

    def load(self, file_path: str) -> List[Tuple[int, str]]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        try:
            if ext == ".pdf":
                return self._load_pdf(file_path)
            elif ext in (".txt", ".md"):
                return self._load_text(file_path)
            elif ext == ".docx":
                return self._load_docx(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def _load_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        logger.info(f"Loaded PDF '{os.path.basename(file_path)}' with {len(pages)} pages.")
        return pages

    def _load_text(self, file_path: str) -> List[Tuple[int, str]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        logger.info(f"Loaded text file '{os.path.basename(file_path)}'.")
        return [(1, content)]

    def _load_docx(self, file_path: str) -> List[Tuple[int, str]]:
        import docx  # python-docx

        doc = docx.Document(file_path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        logger.info(f"Loaded DOCX '{os.path.basename(file_path)}'.")
        return [(1, full_text)]

    def load_folder(self, folder_path: str) -> List[str]:
        """Returns a list of supported file paths inside a folder."""
        files = []
        for root, _, filenames in os.walk(folder_path):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, name))
        logger.info(f"Found {len(files)} supported files in folder '{folder_path}'.")
        return files
