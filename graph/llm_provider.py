"""
Provider abstraction for chat models and embeddings.
"""
from __future__ import annotations

import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


DEFAULT_OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
DEFAULT_OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configured_primary_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _configured_fallback_provider() -> Optional[str]:
    value = os.getenv("FALLBACK_LLM_PROVIDER", "").strip().lower()
    return value or None


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    return "openai" if normalized != "openai" else "openai"


def get_primary_provider() -> str:
    return _normalize_provider(_configured_primary_provider())


def get_fallback_provider() -> Optional[str]:
    value = _configured_fallback_provider()
    if not value:
        return None
    normalized = _normalize_provider(value)
    if normalized == get_primary_provider():
        return None
    return normalized


def create_chat_model(
    *,
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Any:
    provider = _normalize_provider(provider)
    normalized_model = (model or "").strip()
    if normalized_model and "gemini" in normalized_model.lower():
        normalized_model = ""
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}. This project is OpenAI-only.")
    return ChatOpenAI(
        model=normalized_model or DEFAULT_OPENAI_CHAT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_embeddings(provider: Optional[str] = None, model: Optional[str] = None) -> Any:
    provider = _normalize_provider(provider or get_primary_provider())
    normalized_model = (model or "").strip()
    if normalized_model and normalized_model.lower() == "text-embedding-005":
        normalized_model = ""
    if provider != "openai":
        raise ValueError(
            f"Unsupported provider for embeddings: {provider}. This project is OpenAI-only."
        )
    return OpenAIEmbeddings(model=normalized_model or DEFAULT_OPENAI_EMBEDDING_MODEL)

