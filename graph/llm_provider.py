"""
Provider abstraction for chat models and embeddings.
"""
from __future__ import annotations

import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


DEFAULT_OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
DEFAULT_OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_VERTEX_CHAT_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-001")
DEFAULT_VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005")


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_vertex_enabled() -> bool:
    return _is_truthy(os.getenv("ENABLE_VERTEXAI", "true"))


def _configured_primary_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _configured_fallback_provider() -> Optional[str]:
    value = os.getenv("FALLBACK_LLM_PROVIDER", "").strip().lower()
    return value or None


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "vertexai" and not is_vertex_enabled():
        # Prefer explicit fallback provider when Vertex is disabled.
        configured_fallback = _configured_fallback_provider()
        if configured_fallback and configured_fallback != "vertexai":
            return configured_fallback
        return "openai"
    return normalized


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


def _get_vertex_common_kwargs() -> dict:
    project = os.getenv("VERTEX_PROJECT_ID", "").strip()
    location = os.getenv("VERTEX_LOCATION", "").strip() or "us-central1"

    kwargs = {"location": location}
    if project:
        kwargs["project"] = project
    return kwargs


def create_chat_model(
    *,
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Any:
    provider = _normalize_provider(provider)
    normalized_model = (model or "").strip()
    if provider == "openai" and normalized_model and "gemini" in normalized_model.lower():
        normalized_model = ""
    if provider == "vertexai" and normalized_model and normalized_model.lower().startswith("gpt-"):
        normalized_model = ""

    if provider == "openai":
        return ChatOpenAI(
            model=normalized_model or DEFAULT_OPENAI_CHAT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "vertexai":
        from langchain_google_vertexai import ChatVertexAI  # type: ignore[reportMissingImports]

        return ChatVertexAI(
            model_name=normalized_model or DEFAULT_VERTEX_CHAT_MODEL,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **_get_vertex_common_kwargs(),
        )
    raise ValueError(f"Unsupported provider: {provider}")


def create_embeddings(provider: Optional[str] = None, model: Optional[str] = None) -> Any:
    provider = _normalize_provider(provider or get_primary_provider())
    normalized_model = (model or "").strip()
    if provider == "openai" and normalized_model and normalized_model.lower() == "text-embedding-005":
        normalized_model = ""
    if provider == "vertexai" and normalized_model and normalized_model.lower().startswith("text-embedding-3-"):
        normalized_model = ""

    if provider == "openai":
        return OpenAIEmbeddings(model=normalized_model or DEFAULT_OPENAI_EMBEDDING_MODEL)
    if provider == "vertexai":
        return GoogleGenerativeAIEmbeddings(
            model=normalized_model or DEFAULT_VERTEX_EMBEDDING_MODEL,
        )
    raise ValueError(f"Unsupported provider for embeddings: {provider}")

