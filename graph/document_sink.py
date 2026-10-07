"""In-memory document text captured instead of writing the results folder."""

from __future__ import annotations

from typing import Dict, List

_documents: List[Dict[str, str]] = []


def remember_document(file_name: str, document_type: str, content: str) -> None:
    _documents.append(
        {
            "fileName": file_name,
            "documentType": document_type,
            "content": content,
        }
    )


def drain_documents() -> List[Dict[str, str]]:
    docs = list(_documents)
    _documents.clear()
    return docs
