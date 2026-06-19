"""
Console output helpers for the workflow.

Analytics (eval_metrics, node_metrics, status breakdowns, raw agent payloads)
are written to workflow_metrics.jsonl / report files by default, not printed
during agent execution. Set WORKFLOW_CONSOLE_ANALYTICS=true to echo them.

Set WORKFLOW_VERBOSE_CONSOLE=true for debug-level agent/tool dumps.
"""
from __future__ import annotations

import os
from typing import Any, Dict


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def show_console_analytics() -> bool:
    return _env_flag("WORKFLOW_CONSOLE_ANALYTICS")


def show_verbose_console() -> bool:
    return _env_flag("WORKFLOW_VERBOSE_CONSOLE")


def log_progress(message: str) -> None:
    print(message)


def log_analytics(message: str) -> None:
    if show_console_analytics():
        print(message)


def log_verbose(message: str) -> None:
    if show_verbose_console():
        print(message)


def strip_analytics_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal metrics keys before printing results to the console."""
    return {k: v for k, v in payload.items() if k not in ("eval_metrics", "node_metrics")}
