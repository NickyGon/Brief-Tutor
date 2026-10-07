"""Stage labels for a workflow run shown in the frontend."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Callable

StageReporter = Callable[[str, str | None], None]

_reporter: ContextVar[StageReporter | None] = ContextVar("workflow_progress", default=None)

DIAGNOSIS_STAGES: tuple[tuple[str, str], ...] = (
    ("read", "Reading spreadsheet"),
    ("campaigns", "Getting campaigns"),
    ("route", "Choosing the evaluation"),
    ("evaluate", "Evaluating campaigns"),
    ("format", "Writing diagnoses"),
    ("document", "Creating the report"),
    ("finish", "Finishing results"),
)

SIMILARITY_STAGES: tuple[tuple[str, str], ...] = (
    ("read", "Reading spreadsheet"),
    ("campaigns", "Getting campaigns"),
    ("compare", "Comparing similar briefs"),
    ("review", "Reviewing matches"),
    ("report", "Writing the similarity report"),
    ("finish", "Finishing results"),
)


def stages_for(route: str) -> tuple[tuple[str, str], ...]:
    return SIMILARITY_STAGES if route == "similarity" else DIAGNOSIS_STAGES


def initial_progress(route: str) -> dict[str, object]:
    return {
        "id": "start",
        "label": "Starting the workflow",
        "index": 0,
        "total": len(stages_for(route)),
        "percent": 4,
    }


def advance_progress(
    current: dict[str, object] | None,
    route: str,
    stage_id: str,
    detail: str | None = None,
) -> dict[str, object]:
    stages = stages_for(route)
    ids = [stage[0] for stage in stages]
    if stage_id not in ids:
        return current or initial_progress(route)
    index = ids.index(stage_id)
    current_id = None if current is None else current.get("id")
    current_index = -1 if current is None or current_id in (None, "start") else int(current.get("index") or 0)
    if current is not None and current_id not in (None, "start") and index < current_index:
        return current
    label = detail.strip() if isinstance(detail, str) and detail.strip() else stages[index][1]
    total = len(stages)
    percent = max(8, round((index + 0.45) / total * 100))
    if current is not None and current.get("id") == stage_id and current.get("label") == label:
        return current
    return {
        "id": stage_id,
        "label": label,
        "index": index,
        "total": total,
        "percent": percent,
    }


def finished_progress(current: dict[str, object] | None, route: str) -> dict[str, object]:
    base = dict(current or initial_progress(route))
    base["id"] = "done"
    base["label"] = "Run finished"
    base["percent"] = 100
    return base


def bind_progress(callback: StageReporter) -> Token:
    return _reporter.set(callback)


def reset_progress(token: Token) -> None:
    _reporter.reset(token)


def report_stage(stage_id: str, detail: str | None = None) -> None:
    callback = _reporter.get()
    if callback is not None:
        callback(stage_id, detail)
