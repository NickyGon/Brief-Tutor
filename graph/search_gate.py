"""Pause a similarity search until the API caller accepts or declines.

One workflow run is active at a time, so the gate is process-wide.
A context variable can be dropped inside the graph runner; this callback cannot.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

SearchGate = Callable[[Dict[str, str]], bool]
_gate: Optional[SearchGate] = None


def bind_search_gate(callback: SearchGate) -> Optional[SearchGate]:
    global _gate
    previous = _gate
    _gate = callback
    return previous


def reset_search_gate(previous: Optional[SearchGate]) -> None:
    global _gate
    _gate = previous


def ask_search_continue(prompt: Dict[str, str]) -> bool:
    callback = _gate
    if callback is None:
        return False
    return bool(callback(prompt))
