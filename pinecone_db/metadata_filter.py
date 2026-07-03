"""
Small builder utility for constructing Pinecone metadata filter dicts
without hand-writing raw filter syntax everywhere.
"""

from typing import Any, List, Optional


class MetadataFilterBuilder:
    def __init__(self):
        self._conditions = []

    def equals(self, field: str, value: Any) -> "MetadataFilterBuilder":
        self._conditions.append({field: {"$eq": value}})
        return self

    def not_equals(self, field: str, value: Any) -> "MetadataFilterBuilder":
        self._conditions.append({field: {"$ne": value}})
        return self

    def in_list(self, field: str, values: List[Any]) -> "MetadataFilterBuilder":
        self._conditions.append({field: {"$in": values}})
        return self

    def greater_than(self, field: str, value: Any) -> "MetadataFilterBuilder":
        self._conditions.append({field: {"$gt": value}})
        return self

    def less_than(self, field: str, value: Any) -> "MetadataFilterBuilder":
        self._conditions.append({field: {"$lt": value}})
        return self

    def build(self) -> Optional[dict]:
        if not self._conditions:
            return None
        if len(self._conditions) == 1:
            return self._conditions[0]
        return {"$and": self._conditions}


def filter_by_source(source: str) -> dict:
    return MetadataFilterBuilder().equals("source", source).build()


def filter_by_page_range(min_page: int, max_page: int) -> dict:
    return (
        MetadataFilterBuilder()
        .greater_than("page", min_page - 1)
        .less_than("page", max_page + 1)
        .build()
    )
