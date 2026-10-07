"""
Local HTTP API for the Brief Tutor frontend.

Search reads saved workflow artifacts and campaign spreadsheets.
Runs invoke the existing LangGraph workflow and return results only
after that run finishes.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.brief_naming import parse_brief_filename  # noqa: E402
from graph.brief_store import SupabaseUnavailable, load_campaign_brief, open_repository  # noqa: E402
from graph.progress import (  # noqa: E402
    advance_progress,
    bind_progress,
    finished_progress,
    initial_progress,
    reset_progress,
)
from graph.search_gate import bind_search_gate, reset_search_gate  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "credentials",
    ".cursor",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
}
MAX_RESULTS = 40
MAX_EVENTS = 80

app = FastAPI(title="Brief Tutor API", version="1.0.0")


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_active_run_id: str | None = None


class RunRequest(BaseModel):
    spreadsheet_path: str = Field(default="", alias="spreadsheetPath")
    route: str = "diagnosis"

    model_config = {"populate_by_name": True}


def _route_value(route: str) -> str:
    normalized = route.strip().lower()
    if normalized in {"similarity", "similar", "0"}:
        return "0"
    if normalized in {"diagnosis", "evaluate", "1"}:
        return "1"
    raise HTTPException(status_code=400, detail="route must be diagnosis or similarity.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel_id(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _resolve_id(artifact_id: str) -> Path:
    raw = (artifact_id or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or ".." in Path(raw).parts:
        raise HTTPException(status_code=400, detail="Invalid artifact id.")
    path = (PROJECT_ROOT / raw).resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="Artifact is outside the project.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return path


def _resolve_spreadsheet(spreadsheet_path: str) -> Path:
    raw = (spreadsheet_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="spreadsheetPath is required.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    path = candidate.resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="Spreadsheet is outside the project.")
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise HTTPException(status_code=400, detail="Spreadsheet must be an .xlsx file.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Spreadsheet not found: {_rel_id(path)}")
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _similarity_body(data: dict[str, Any]) -> dict[str, Any] | None:
    nested = data.get("family_similarity")
    if isinstance(nested, dict):
        return nested
    if any(key in data for key in ("absolute_pairs", "likely_pairs", "review_pairs", "candidate_files")):
        return data
    return None


def _kind_for(path: Path, data: dict[str, Any] | None) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "brief"
    return None


def _status_breakdown(diagnoses: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"critical": 0, "observed": 0, "passed": 0}
    for item in diagnoses:
        status = str(item.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _pair(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    score = item.get("similarity_score", item.get("similarity_percent"))
    try:
        score_value = float(score or 0)
    except (TypeError, ValueError):
        score_value = 0.0
    if score_value > 1:
        score_value = score_value / 100

    def _unit(key: str) -> float:
        try:
            value = float(item.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0
        return value / 100 if value > 1 else value

    evidence = item.get("evidence_points") or item.get("evidencePoints") or []
    return {
        "targetCampaignId": str(item.get("target_campaign_id") or item.get("targetCampaignId") or ""),
        "candidateCampaignId": str(item.get("candidate_campaign_id") or item.get("candidateCampaignId") or ""),
        "fileName": str(item.get("file_name") or item.get("fileName") or ""),
        "similarityScore": score_value,
        "pairStatus": str(item.get("pair_status") or item.get("pairStatus") or ""),
        "pairBasis": str(item.get("pair_basis") or item.get("pairBasis") or ""),
        "scoringPath": str(item.get("scoring_path") or item.get("scoringPath") or ""),
        "matchReason": str(item.get("match_reason") or item.get("matchReason") or ""),
        "evidencePoints": [str(point) for point in evidence if str(point).strip()][:6],
        "styleDirectionSimilarity": _unit("style_direction_similarity"),
        "campaignWordingSimilarity": _unit("campaign_wording_similarity"),
        "dealershipRelationship": _unit("dealership_relationship"),
    }


def _diagnosis(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    return {
        "campaignId": str(item.get("campaign_id") or item.get("campaignId") or item.get("campaign_external_id") or ""),
        "status": str(item.get("status") or "observed"),
        "diagnosis": str(item.get("diagnosis") or ""),
        "issues": [str(value) for value in _as_list(item.get("issues"))],
        "recommendations": [str(value) for value in _as_list(item.get("recommendations"))],
        "groundingEvidence": [
            str(value) for value in _as_list(item.get("grounding_evidence") or item.get("groundingEvidence"))
        ],
    }


def _unit_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1:
        score = score / 100
    return score


def _candidate(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    try:
        score = float(item.get("file_similarity_score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    if score > 1:
        score = score / 100
    return {
        "fileName": str(item.get("file_name") or Path(str(item.get("file_path") or "")).name),
        "dealershipName": item.get("dealership_name"),
        "taskType": item.get("task_type"),
        "fileSimilarityScore": score,
        "briefSimilarityScore": _unit_score(item.get("brief_similarity_score")),
        "matchKind": item.get("match_kind"),
        "coverage": _unit_score(item.get("coverage")),
    }


def _parse_name(path: Path) -> dict[str, str]:
    name = path.name
    for suffix in ("-diagnoses.json", "-family-similarity.json", ".json", ".xlsx", ".xlsm"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return parse_brief_filename(f"{name}.xlsx") or {}


def _normalize(path: Path, data: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    parsed = _parse_name(path)
    payload = data or {}
    similarity = _similarity_body(payload) or {}
    diagnoses = [_diagnosis(item) for item in _as_list(payload.get("campaign_diagnoses"))]
    diagnoses = [item for item in diagnoses if item]
    breakdown = payload.get("status_breakdown") if isinstance(payload.get("status_breakdown"), dict) else None
    if diagnoses and not breakdown:
        breakdown = _status_breakdown([item for item in _as_list(payload.get("campaign_diagnoses")) if isinstance(item, dict)])

    dealership = payload.get("dealership_name") or similarity.get("dealership_name")
    task_type = payload.get("task_type") or similarity.get("task_type")
    spreadsheet = payload.get("spreadsheet_path") or similarity.get("target_file_path")
    if kind == "brief":
        spreadsheet = str(path)

    eval_metrics = payload.get("eval_metrics") if isinstance(payload.get("eval_metrics"), dict) else None
    eval_view = None
    if isinstance(eval_metrics, dict) and eval_metrics:
        eval_view = {
            "groundednessScore": eval_metrics.get("groundedness_score"),
            "fieldCompletenessScore": eval_metrics.get("field_completeness_score"),
            "passed": eval_metrics.get("passed"),
        }

    similarity_view = None
    if kind == "similarity" or similarity:
        if similarity:
            similarity_view = {
                "targetFileName": str(similarity.get("target_file_name") or path.name),
                "warnings": [str(item) for item in _as_list(similarity.get("warnings"))],
                "absolutePairs": [pair for pair in (_pair(item) for item in _as_list(similarity.get("absolute_pairs"))) if pair],
                "likelyPairs": [pair for pair in (_pair(item) for item in _as_list(similarity.get("likely_pairs"))) if pair],
                "reviewPairs": [pair for pair in (_pair(item) for item in _as_list(similarity.get("review_pairs") or similarity.get("review_matches"))) if pair],
                "unpairedTargets": [str(item) for item in _as_list(similarity.get("unpaired_targets"))],
                "candidateFiles": [item for item in (_candidate(item) for item in _as_list(similarity.get("candidate_files"))) if item][:12],
                "briefSimilarityScore": None,
                "matchKind": None,
            }
            scored = [
                item for item in similarity_view["candidateFiles"]
                if item.get("briefSimilarityScore") is not None
            ]
            if scored:
                best = max(scored, key=lambda item: float(item.get("briefSimilarityScore") or 0))
                similarity_view["briefSimilarityScore"] = best.get("briefSimilarityScore")
                similarity_view["matchKind"] = best.get("matchKind")

    title = path.name
    if kind == "similarity" and similarity.get("target_file_name"):
        title = str(similarity["target_file_name"])
    headline_bits: list[str] = []
    if task_type:
        headline_bits.append(str(task_type))
    if dealership:
        headline_bits.append(str(dealership))
    if kind == "diagnosis" and breakdown:
        headline_bits.append(
            f"{breakdown.get('critical', 0)} critical · {breakdown.get('observed', 0)} observed · {breakdown.get('passed', 0)} passed"
        )
    if similarity_view:
        headline_bits.append(
            f"{len(similarity_view['absolutePairs'])} absolute · {len(similarity_view['likelyPairs'])} likely · {len(similarity_view['reviewPairs'])} review"
        )
    if kind == "brief" and parsed:
        headline_bits.append(f"{parsed.get('year')}-{parsed.get('month')} · {parsed.get('campaign_token')}")

    updated = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "id": _rel_id(path) if path.is_relative_to(PROJECT_ROOT) else path.name,
        "kind": "similarity" if similarity_view and kind != "diagnosis" and kind != "brief" else kind,
        "title": title,
        "dealershipName": dealership,
        "taskType": task_type,
        "briefId": parsed.get("brief_id"),
        "accountId": parsed.get("account_id"),
        "campaignToken": parsed.get("campaign_token"),
        "updatedAt": updated,
        "headline": " · ".join(headline_bits) if headline_bits else path.suffix.lstrip(".").upper(),
        "spreadsheetPath": str(spreadsheet) if spreadsheet else (str(path) if kind == "brief" else None),
        "totalCampaigns": payload.get("total_campaigns"),
        "diagnoses": diagnoses,
        "statusBreakdown": breakdown,
        "eval": eval_view,
        "similarity": similarity_view if kind != "brief" else None,
        "searchText": " ".join(
            [
                path.name,
                str(dealership or ""),
                str(task_type or ""),
                str(parsed.get("account_id") or ""),
                str(parsed.get("campaign_token") or ""),
                str(parsed.get("brief_id") or ""),
                " ".join(item["campaignId"] + " " + item["status"] + " " + item["diagnosis"] for item in diagnoses),
                " ".join(
                    pair["targetCampaignId"] + " " + pair["candidateCampaignId"] + " " + pair["fileName"] + " " + pair["matchReason"]
                    for pair in (
                        (similarity_view or {}).get("absolutePairs", [])
                        + (similarity_view or {}).get("likelyPairs", [])
                        + (similarity_view or {}).get("reviewPairs", [])
                    )
                ),
            ]
        ).lower(),
    }


def _hit(record: dict[str, Any]) -> dict[str, Any]:
    similarity = record.get("similarity") or {}
    return {
        "id": record["id"],
        "kind": record["kind"],
        "title": record["title"],
        "dealershipName": record.get("dealershipName"),
        "taskType": record.get("taskType"),
        "briefId": record.get("briefId"),
        "updatedAt": record["updatedAt"],
        "headline": record["headline"],
        "statusBreakdown": record.get("statusBreakdown"),
        "pairCounts": {
            "absolute": len(similarity.get("absolutePairs") or []),
            "likely": len(similarity.get("likelyPairs") or []),
            "review": len(similarity.get("reviewPairs") or []),
            "unpaired": len(similarity.get("unpairedTargets") or []),
        }
        if similarity
        else None,
    }


def _public_detail(record: dict[str, Any]) -> dict[str, Any]:
    detail = dict(record)
    detail.pop("searchText", None)
    spreadsheet = detail.get("spreadsheetPath")
    if isinstance(spreadsheet, str):
        path = Path(spreadsheet)
        try:
            if path.is_absolute() and PROJECT_ROOT in path.resolve().parents:
                detail["spreadsheetPath"] = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except OSError:
            pass
    return detail


def _iter_artifact_files() -> list[Path]:
    found: list[Path] = []
    for root, dirs, files in os_walk_sorted():
        for name in files:
            lower = name.lower()
            if lower.endswith((".xlsx", ".xlsm", ".json")):
                found.append(Path(root) / name)
    return found


def os_walk_sorted():
    import os

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        yield root, dirs, files


def _collect(kind_filter: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _iter_artifact_files():
        data = _read_json(path) if path.suffix.lower() == ".json" else None
        kind = _kind_for(path, data)
        if kind is None:
            continue
        if kind_filter != "all" and kind != kind_filter:
            continue
        try:
            records.append(_normalize(path, data, kind))
        except OSError:
            continue
    records.sort(key=lambda item: item.get("updatedAt") or "", reverse=True)
    return records


def _detail_from_results(final_results: dict[str, Any], spreadsheet: Path) -> dict[str, Any]:
    mode = str(final_results.get("mode") or "")
    kind = "similarity" if mode == "family_similarity" or "family_similarity" in final_results else "diagnosis"
    record = _normalize(spreadsheet, final_results, kind)
    record["id"] = f"run:{spreadsheet.name}"
    record["kind"] = kind
    record["title"] = spreadsheet.name
    record["spreadsheetPath"] = _rel_id(spreadsheet)
    record["updatedAt"] = _now()
    return _public_detail(record)


def _write_console(value: str) -> None:
    stream = sys.__stdout__
    if stream is None:
        return
    try:
        stream.write(value)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "cp1252"
        stream.write(value.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _append_event(job: dict[str, Any], message: str) -> None:
    text = message.strip()
    if not text:
        return
    events = job.setdefault("events", [])
    events.append({"at": _now(), "message": text[:500]})
    if len(events) > MAX_EVENTS:
        del events[: len(events) - MAX_EVENTS]


def _execute_run(job_id: str, spreadsheet: Path, route_value: str) -> None:
    global _active_run_id
    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = "running"
        job["startedAt"] = _now()
    route_label = "similarity" if route_value == "0" else "diagnosis"
    _append_event(job, f"Workflow started on the {route_label} route.")
    previous_route = os.environ.get("BRIEF_POST_PARSE_ROUTE")
    os.environ["BRIEF_POST_PARSE_ROUTE"] = route_value

    class _Progress:
        encoding = "utf-8"

        def write(self, value: str) -> int:
            _write_console(value)
            for line in value.splitlines():
                _append_event(job, line)
            return len(value)

        def flush(self) -> None:
            sys.__stdout__.flush()

        def isatty(self) -> bool:
            return False

    previous_stdout = sys.stdout
    sys.stdout = _Progress()  # type: ignore[assignment]

    def _on_stage(stage_id: str, detail: str | None = None) -> None:
        with _jobs_lock:
            job["progress"] = advance_progress(job.get("progress"), route_label, stage_id, detail)

    def _on_search_prompt(prompt: dict[str, str]) -> bool:
        event = threading.Event()
        with _jobs_lock:
            job["status"] = "awaiting_input"
            job["prompt"] = {
                "question": prompt.get("question") or "Search for more similar briefs?",
                "candidateName": prompt.get("candidateName") or "",
                "nextScope": prompt.get("nextScope") or "",
            }
            job["_continue_event"] = event
        event.wait()
        with _jobs_lock:
            accept = bool(job.pop("_continue_accept", False))
            job["status"] = "running"
            job["prompt"] = None
            job.pop("_continue_event", None)
        return accept

    progress_token = bind_progress(_on_stage)
    search_token = bind_search_gate(_on_search_prompt)
    try:
        from graph.langsmith_workflow import run_campaign_brief_workflow_traced
        from graph.models import AgentState

        initial_state = AgentState(
            messages=[
                {
                    "role": "user",
                    "content": f"Please process the campaign brief spreadsheet at: {spreadsheet}",
                }
            ],
            next_node="brief_creator",
        )
        final_state = run_campaign_brief_workflow_traced(
            initial_state.model_dump(),
            str(spreadsheet),
        )
        final_results = final_state.get("final_results") if isinstance(final_state, dict) else None
        if hasattr(final_results, "model_dump"):
            final_results = final_results.model_dump()
        if not isinstance(final_results, dict):
            raise RuntimeError("Workflow finished without final_results.")
        display = {
            key: value
            for key, value in final_results.items()
            if key not in {"node_metrics", "db"}
        }
        with _jobs_lock:
            job["status"] = "completed"
            job["finishedAt"] = _now()
            job["progress"] = finished_progress(job.get("progress"), route_label)
            job["result"] = _detail_from_results(display, spreadsheet)
        _append_event(job, "Workflow completed.")
    except Exception as exc:
        with _jobs_lock:
            job["status"] = "failed"
            job["finishedAt"] = _now()
            job["error"] = str(exc)
        _append_event(job, f"Workflow failed: {exc}")
        traceback.print_exc()
    finally:
        reset_progress(progress_token)
        reset_search_gate(search_token)
        sys.stdout = previous_stdout
        if previous_route is None:
            os.environ.pop("BRIEF_POST_PARSE_ROUTE", None)
        else:
            os.environ["BRIEF_POST_PARSE_ROUTE"] = previous_route
        with _jobs_lock:
            if _active_run_id == job_id:
                _active_run_id = None


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "spreadsheetPath": job["spreadsheetPath"],
        "route": job.get("route"),
        "startedAt": job.get("startedAt"),
        "finishedAt": job.get("finishedAt"),
        "error": job.get("error"),
        "progress": job.get("progress"),
        "events": list(job.get("events") or []),
        "result": job.get("result"),
        "prompt": job.get("prompt"),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "brief-tutor"}


@app.get("/api/search")
def search(q: str = "", kind: str = "all") -> dict[str, Any]:
    started = time.perf_counter()
    query = q.strip().lower()
    kind_filter = kind if kind in {"all", "diagnosis", "similarity", "brief"} else "all"
    tokens = [token for token in query.split() if token]
    matched: list[dict[str, Any]] = []
    for record in _collect(kind_filter):
        haystack = record.get("searchText") or ""
        if tokens and not all(token in haystack for token in tokens):
            continue
        matched.append(record)
        if len(matched) >= MAX_RESULTS:
            break
    took_ms = int((time.perf_counter() - started) * 1000)
    return {
        "query": q,
        "kind": kind_filter,
        "tookMs": took_ms,
        "resultCount": len(matched),
        "results": [_hit(record) for record in matched],
    }


@app.get("/api/artifacts")
def artifact(id: str) -> dict[str, Any]:
    path = _resolve_id(id)
    data = _read_json(path) if path.suffix.lower() == ".json" else None
    kind = _kind_for(path, data)
    if kind is None:
        raise HTTPException(status_code=404, detail="Artifact is not a workflow result.")
    return _public_detail(_normalize(path, data, kind))


@app.post("/api/runs", status_code=202)
def start_run(body: RunRequest) -> dict[str, Any]:
    global _active_run_id
    spreadsheet = _resolve_spreadsheet(body.spreadsheet_path)
    route_value = _route_value(body.route)
    route_name = "similarity" if route_value == "0" else "diagnosis"
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "spreadsheetPath": _rel_id(spreadsheet),
        "route": route_name,
        "startedAt": None,
        "finishedAt": None,
        "error": None,
        "progress": initial_progress(route_name),
        "events": [{"at": _now(), "message": "Queued."}],
        "result": None,
        "prompt": None,
    }
    with _jobs_lock:
        if _active_run_id is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "A workflow run is already in progress.",
                    "activeRunId": _active_run_id,
                },
            )
        _active_run_id = job_id
        _jobs[job_id] = job
    thread = threading.Thread(target=_execute_run, args=(job_id, spreadsheet, route_value), daemon=True)
    thread.start()
    return _job_snapshot(job)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return _job_snapshot(job)


class ContinueRequest(BaseModel):
    accept: bool = False


@app.post("/api/runs/{run_id}/continue")
def continue_run(run_id: str, body: ContinueRequest) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if job.get("status") != "awaiting_input":
            raise HTTPException(status_code=409, detail="This run is not waiting for a decision.")
        job["_continue_accept"] = body.accept
        event = job.get("_continue_event")
    if event is not None:
        event.set()
    return _job_snapshot(job)


def _campaign_view(row: dict[str, Any]) -> dict[str, Any]:
    style = row.get("style_descriptions") if isinstance(row.get("style_descriptions"), dict) else {}
    assets = row.get("assets") if isinstance(row.get("assets"), dict) else {}
    return {
        "campaignId": str(row.get("campaign_external_id") or ""),
        "offer": {
            "headline": str(row.get("headline") or ""),
            "offer": str(row.get("offer") or ""),
            "body": str(row.get("body") or ""),
            "cta": str(row.get("cta") or ""),
            "disclaimer": str(row.get("disclaimer") or ""),
        },
        "style": {
            "direction": str(style.get("asset_style_direction") or ""),
            "additional": str(style.get("additional_style_information") or ""),
            "vehiclePhotography": str(style.get("vehicle_photography") or ""),
            "logos": str(style.get("logos") or ""),
        },
        "assets": {
            "desktop": str(assets.get("sl_bn_srp_da") or ""),
            "mobile": str(assets.get("sl_m_bn_m") or ""),
            "facebook": str(assets.get("facebook_assets") or ""),
            "instagram": str(assets.get("instagram_assets") or ""),
            "google": str(assets.get("google_assets") or ""),
            "ot1": str(assets.get("ot_1") or ""),
            "ot2": str(assets.get("ot_2") or ""),
            "ot3": str(assets.get("ot_3") or ""),
            "ot4": str(assets.get("ot_4") or ""),
            "ot5": str(assets.get("ot_5") or ""),
            "ot6": str(assets.get("ot_6") or ""),
        },
    }


def _empty_history(message: str | None = None, supabase_error: str | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "supabaseError": supabase_error,
        "message": message,
        "dealershipName": None,
        "taskType": None,
        "accountId": None,
        "campaigns": [],
        "diagnoses": [],
        "similarity": None,
    }


@app.get("/api/briefs/history")
def brief_history(spreadsheetPath: str) -> dict[str, Any]:
    spreadsheet = _resolve_spreadsheet(spreadsheetPath)
    try:
        repo = open_repository()
    except SupabaseUnavailable as exc:
        return _empty_history(supabase_error=str(exc))
    if repo is None:
        return _empty_history(message="Supabase is not configured.")
    from graph.supabase.repository import parse_brief_identifiers

    ids = parse_brief_identifiers(str(spreadsheet))
    brief_id = str(ids.get("brief_id") or ids.get("instance_id") or spreadsheet.stem)
    try:
        row = repo.get_brief_by_brief_id(brief_id)
        if not isinstance(row, dict):
            return _empty_history(message="This brief is not stored in Supabase yet.")
        brief_fk = row.get("id")
        campaigns = repo.list_campaigns_by_brief_fk(brief_fk) if isinstance(brief_fk, int) else []
        diagnoses = repo.list_latest_diagnoses(brief_fk) if isinstance(brief_fk, int) else []
        try:
            matches = repo.list_brief_similarity_matches(brief_id.upper())
        except Exception:
            matches = []
    except SupabaseUnavailable:
        raise
    except Exception as exc:
        return _empty_history(supabase_error=f"Supabase could not be consulted: {exc}")

    pairs: list[dict[str, Any]] = []
    candidate_files: list[dict[str, Any]] = []
    best_score = None
    best_kind = None
    for match in matches:
        score = float(match.get("brief_similarity_score") or 0)
        kind = str(match.get("match_kind") or "none")
        if best_score is None or score > best_score:
            best_score = score
            best_kind = kind
        payload = match.get("pair_payload") if isinstance(match.get("pair_payload"), dict) else {}
        candidate_files.append(
            {
                "fileName": match.get("candidate_file_name") or "",
                "dealershipName": None,
                "taskType": None,
                "fileSimilarityScore": score,
                "briefSimilarityScore": score,
                "matchKind": kind,
                "coverage": float(match.get("coverage") or 0),
            }
        )
        candidate_name = str(match.get("candidate_file_name") or "")
        for key, status in (("absolute_pairs", "absolute"), ("likely_pairs", "likely"), ("review_pairs", "review")):
            for pair in payload.get(key) or []:
                if isinstance(pair, dict):
                    pair = {
                        **pair,
                        "pair_status": pair.get("pair_status") or status,
                        "file_name": pair.get("file_name") or candidate_name,
                    }
                    normalized = _pair(pair)
                    if normalized:
                        pairs.append(normalized)
    similarity = None
    if matches:
        similarity = {
            "targetFileName": spreadsheet.name,
            "warnings": [],
            "absolutePairs": [item for item in pairs if item["pairStatus"] == "absolute"],
            "likelyPairs": [item for item in pairs if item["pairStatus"] == "likely"],
            "reviewPairs": [item for item in pairs if item["pairStatus"] == "review"],
            "unpairedTargets": [],
            "candidateFiles": candidate_files,
            "briefSimilarityScore": best_score,
            "matchKind": best_kind,
        }
    return {
        "available": True,
        "supabaseError": None,
        "dealershipName": row.get("dealership_name"),
        "taskType": row.get("task_type"),
        "accountId": ids.get("account_id"),
        "campaigns": [_campaign_view(item) for item in campaigns if isinstance(item, dict)],
        "diagnoses": [
            _diagnosis(item)
            for item in diagnoses
            if _diagnosis(item)
        ],
        "similarity": similarity,
    }


class RetrieveRequest(BaseModel):
    model_config = {"populate_by_name": True}
    spreadsheet_path: str = Field(default="", alias="spreadsheetPath")


@app.post("/api/briefs/retrieve")
def retrieve_brief(body: RetrieveRequest) -> dict[str, Any]:
    spreadsheet = _resolve_spreadsheet(body.spreadsheet_path)
    try:
        repo = open_repository()
    except SupabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if repo is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    loaded = load_campaign_brief(str(spreadsheet))
    if loaded is None:
        from graph.tools import parse_local_spreadsheet_to_campaign_brief
        from graph.workflow import _persist_brief_and_campaigns

        loaded = parse_local_spreadsheet_to_campaign_brief(str(spreadsheet))
        _persist_brief_and_campaigns(loaded, {}, create_run_if_missing=False)
    from graph.supabase.repository import parse_brief_identifiers
    return {
        "available": True,
        "brief": {
            "spreadsheetPath": loaded.spreadsheet_path,
            "taskType": loaded.task_type,
            "assetSummary": loaded.asset_summary,
            "dealershipName": loaded.dealership_name,
            "accountId": parse_brief_identifiers(loaded.spreadsheet_path).get("dealership_family_id"),
            "content11To20": loaded.content_11_20,
            "campaigns": [
                _campaign_view(
                    {
                        "campaign_external_id": campaign.campaign_id,
                        "headline": campaign.offer_details.headline,
                        "offer": campaign.offer_details.offer,
                        "body": campaign.offer_details.body,
                        "cta": campaign.offer_details.cta,
                        "disclaimer": campaign.offer_details.disclaimer,
                        "style_descriptions": campaign.style_descriptions.model_dump(mode="python"),
                        "assets": campaign.assets.model_dump(mode="python"),
                    }
                )
                for campaign in loaded.campaigns
            ],
        },
    }
