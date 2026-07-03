"""
Generic helper functions used across the project.
"""

import hashlib
import json
import os
import time
import uuid
from functools import wraps
from typing import Callable

from utils.logger import get_logger

logger = get_logger("helper")


def compute_file_hash(file_path: str) -> str:
    """SHA-256 hash of a file's content, used for duplicate detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def retry(max_attempts: int = 3, delay_seconds: float = 1.5, backoff: float = 2.0):
    """Simple retry decorator with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay_seconds
            last_exc = None
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        f"[Retry {attempt}/{max_attempts}] {func.__name__} failed: {e}"
                    )
                    if attempt == max_attempts:
                        break
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            logger.error(f"{func.__name__} failed after {max_attempts} attempts.")
            raise last_exc

        return wrapper

    return decorator


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def truncate(text: str, max_chars: int = 200) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
