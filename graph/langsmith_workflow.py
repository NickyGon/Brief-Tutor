"""
LangSmith-friendly tracing for the campaign brief LangGraph workflow.

Provides a root trace with readable inputs/outputs (via process_inputs/process_outputs)
and passes run_name, tags, and metadata into the graph invoke config for the LangGraph run.

Optional: append the same compact inputs/outputs as a row in a LangSmith **Key-value**
dataset (set LANGSMITH_DATASET_NAME or LANGSMITH_DATASET_ID in the environment).
"""
from __future__ import annotations

import logging
import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from graph.workflow import create_campaign_workflow

logger = logging.getLogger(__name__)


def _workflow_run_config(spreadsheet_path: str) -> Dict[str, Any]:
    filename = Path(spreadsheet_path).name
    return {
        "run_name": f"campaign_brief:{filename}",
        "tags": ["brief-tutor", "campaign_brief", "langgraph"],
        "metadata": {
            "workflow": "campaign_brief",
            "spreadsheet_path": spreadsheet_path,
            "spreadsheet_filename": filename,
        },
    }


def _compact_trace_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Shrink state dict for LangSmith Inputs (full messages can be huge)."""
    state = inputs.get("initial_state_dict") or {}
    if not isinstance(state, dict):
        return {"initial_state_dict": repr(state)[:500]}

    messages: List[Any] = state.get("messages") or []
    user_preview: Optional[str] = None
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            user_preview = (content[:800] + "…") if isinstance(content, str) and len(content) > 800 else content
            break

    return {
        "spreadsheet_path": inputs.get("spreadsheet_path"),
        "next_node": state.get("next_node"),
        "message_count": len(messages),
        "initial_user_message_preview": user_preview,
        "state_metadata_keys": sorted((state.get("metadata") or {}).keys()) if isinstance(state.get("metadata"), dict) else [],
    }


def _brief_summary(brief: Any) -> Dict[str, Any]:
    if brief is None:
        return {}
    if hasattr(brief, "task_type"):
        return {
            "task_type": brief.task_type,
            "dealership_name": getattr(brief, "dealership_name", None),
            "campaign_count": len(brief.campaigns) if getattr(brief, "campaigns", None) is not None else None,
            "spreadsheet_path": getattr(brief, "spreadsheet_path", None),
        }
    if isinstance(brief, dict):
        camps = brief.get("campaigns") or []
        return {
            "task_type": brief.get("task_type"),
            "dealership_name": brief.get("dealership_name"),
            "campaign_count": len(camps) if isinstance(camps, list) else None,
            "spreadsheet_path": brief.get("spreadsheet_path"),
        }
    return {"brief_repr": repr(brief)[:300]}


def _diagnosis_preview(diagnoses: Any, limit: int = 25) -> List[Dict[str, Any]]:
    if not diagnoses or not isinstance(diagnoses, list):
        return []
    out: List[Dict[str, Any]] = []
    for d in diagnoses[:limit]:
        if hasattr(d, "model_dump"):
            d = d.model_dump(mode="python")
        if isinstance(d, dict):
            out.append(
                {
                    "campaign_id": d.get("campaign_id"),
                    "status": d.get("status"),
                }
            )
    return out


def _compact_trace_outputs(outputs: Any) -> Dict[str, Any]:
    """Summarize final graph state for LangSmith Outputs."""
    if not isinstance(outputs, dict):
        return {"output": repr(outputs)[:2000]}

    brief_summary = _brief_summary(outputs.get("campaign_brief"))
    final_results = outputs.get("final_results")
    result_summary: Dict[str, Any] = {}
    if isinstance(final_results, dict):
        result_summary = {
            "final_results_keys": sorted(final_results.keys()),
        }
        if "task_type" in final_results:
            result_summary["final_task_type"] = final_results.get("task_type")
        if "campaigns" in final_results and isinstance(final_results["campaigns"], list):
            result_summary["final_campaign_count"] = len(final_results["campaigns"])

    return {
        "state_keys": sorted(outputs.keys()),
        "next": outputs.get("next"),
        "next_node": outputs.get("next_node"),
        "rework_count": outputs.get("rework_count"),
        "qa_result": outputs.get("qa_result"),
        "qa_feedback_preview": (
            (outputs.get("qa_feedback") or "")[:400] + "…"
            if isinstance(outputs.get("qa_feedback"), str) and len(outputs.get("qa_feedback") or "") > 400
            else outputs.get("qa_feedback")
        ),
        "diagnoses_json_path": outputs.get("diagnoses_json_path"),
        "diagnosis_count": len(outputs["campaign_diagnoses"]) if outputs.get("campaign_diagnoses") else 0,
        "diagnoses_preview": _diagnosis_preview(outputs.get("campaign_diagnoses")),
        "eval_metrics": ((outputs.get("metadata") or {}).get("eval_metrics") if isinstance(outputs.get("metadata"), dict) else {}),
        "node_metrics": ((outputs.get("metadata") or {}).get("node_metrics") if isinstance(outputs.get("metadata"), dict) else []),
        "campaign_brief_summary": brief_summary,
        "has_final_results": final_results is not None,
        "final_results_summary": result_summary,
    }


def _metrics_provider_env() -> Dict[str, Any]:
    """Non-secret provider configuration for metrics reports and downstream tooling."""
    return {
        "LLM_PROVIDER": (os.getenv("LLM_PROVIDER") or "").strip() or None,
        "FALLBACK_LLM_PROVIDER": (os.getenv("FALLBACK_LLM_PROVIDER") or "").strip() or None,
        "EMBEDDING_PROVIDER": (os.getenv("EMBEDDING_PROVIDER") or "").strip() or None,
        "OPENAI_CHAT_MODEL": (os.getenv("OPENAI_CHAT_MODEL") or "").strip() or None,
        "LLM_MODEL": (os.getenv("LLM_MODEL") or "").strip() or None,
        "FALLBACK_LLM_MODEL": (os.getenv("FALLBACK_LLM_MODEL") or "").strip() or None,
    }


def _rollup_node_metrics(node_metrics: Any) -> Dict[str, Any]:
    """Summarize per-node LLM metrics (provider, latency, fallback) for report consumers."""
    if not isinstance(node_metrics, list):
        return {}

    by_provider: Dict[str, Dict[str, Any]] = {}
    fallback_invocations = 0

    for m in node_metrics:
        if not isinstance(m, dict):
            continue
        prov = str(m.get("provider") or "unknown")
        if m.get("fallback_used"):
            fallback_invocations += 1
        bucket = by_provider.setdefault(
            prov,
            {"invocations": 0, "latency_ms_total": 0, "nodes": []},
        )
        bucket["invocations"] += 1
        lat = m.get("latency_ms")
        if isinstance(lat, (int, float)):
            bucket["latency_ms_total"] += int(lat)
        node_name = m.get("node")
        if node_name is not None:
            bucket["nodes"].append(node_name)

    for pdata in by_provider.values():
        inv = pdata.get("invocations") or 0
        total = pdata.get("latency_ms_total") or 0
        pdata["latency_ms_avg"] = round(total / inv, 2) if inv else 0.0

    return {
        "total_node_invocations": len([m for m in node_metrics if isinstance(m, dict)]),
        "fallback_invocations": fallback_invocations,
        "by_invoked_provider": by_provider,
    }


def _final_results_metrics_slice(final_state: Dict[str, Any]) -> Dict[str, Any]:
    fr = final_state.get("final_results")
    if not isinstance(fr, dict):
        return {}
    return {
        k: fr.get(k)
        for k in ("task_type", "total_campaigns", "status_breakdown", "rework_count")
        if k in fr
    }


def _emit_regression_alerts(final_state: Dict[str, Any]) -> List[str]:
    alerts: List[str] = []
    metadata = final_state.get("metadata") if isinstance(final_state, dict) else {}
    if not isinstance(metadata, dict):
        return alerts

    eval_metrics = metadata.get("eval_metrics") or {}
    if isinstance(eval_metrics, dict):
        groundedness = eval_metrics.get("groundedness_score")
        min_groundedness = eval_metrics.get("min_groundedness")
        if isinstance(groundedness, (int, float)) and isinstance(min_groundedness, (int, float)) and groundedness < min_groundedness:
            alerts.append(f"groundedness below threshold ({groundedness:.3f} < {min_groundedness:.3f})")

        completeness = eval_metrics.get("field_completeness_score")
        min_completeness = eval_metrics.get("min_field_completeness")
        if isinstance(completeness, (int, float)) and isinstance(min_completeness, (int, float)) and completeness < min_completeness:
            alerts.append(f"field completeness below threshold ({completeness:.3f} < {min_completeness:.3f})")

    node_metrics = metadata.get("node_metrics") or []
    if isinstance(node_metrics, list):
        for node_metric in node_metrics:
            if not isinstance(node_metric, dict):
                continue
            latency_ms = node_metric.get("latency_ms")
            node = node_metric.get("node", "unknown")
            if isinstance(latency_ms, int) and latency_ms > int(os.getenv("ALERT_MAX_NODE_LATENCY_MS", "45000")):
                alerts.append(f"high latency detected on {node} ({latency_ms}ms)")
    return alerts


def _append_workflow_metrics(spreadsheet_path: str, final_state: Dict[str, Any], workflow_latency_ms: int) -> None:
    metrics_path = Path(os.getenv("WORKFLOW_METRICS_FILE", "workflow_metrics.jsonl"))
    metadata = final_state.get("metadata") if isinstance(final_state, dict) else {}
    record = {
        "workflow": "campaign_brief",
        "spreadsheet_path": spreadsheet_path,
        "latency_ms": workflow_latency_ms,
        "eval_metrics": (metadata or {}).get("eval_metrics", {}) if isinstance(metadata, dict) else {},
        "node_metrics": (metadata or {}).get("node_metrics", []) if isinstance(metadata, dict) else [],
        "alerts": _emit_regression_alerts(final_state),
    }
    try:
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Could not append workflow metrics: %s", exc)


def _write_workflow_metrics_report(
    spreadsheet_path: str,
    final_state: Dict[str, Any],
    workflow_latency_ms: int,
) -> None:
    """
    Write a single-run JSON report (eval + node/provider metrics) for dashboards or other pipelines.
    Set WORKFLOW_METRICS_REPORT_FILE to a path (e.g. reports/last_run_metrics.json).
    """
    report_path_raw = (os.getenv("WORKFLOW_METRICS_REPORT_FILE") or "").strip()
    if not report_path_raw:
        return

    report_path = Path(report_path_raw)
    metadata = final_state.get("metadata") if isinstance(final_state, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    eval_metrics = metadata.get("eval_metrics") or {}
    if not isinstance(eval_metrics, dict):
        eval_metrics = {}

    node_metrics = metadata.get("node_metrics") or []
    if not isinstance(node_metrics, list):
        node_metrics = []

    alerts = _emit_regression_alerts(final_state)
    report = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow": "campaign_brief",
        "spreadsheet_path": spreadsheet_path,
        "workflow_latency_ms": workflow_latency_ms,
        "provider_config": _metrics_provider_env(),
        "eval_metrics": eval_metrics,
        "node_metrics": node_metrics,
        "node_metrics_summary": _rollup_node_metrics(node_metrics),
        "alerts": alerts,
        "final_results_summary": _final_results_metrics_slice(final_state),
        "qa_result": final_state.get("qa_result") if isinstance(final_state, dict) else None,
    }

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except Exception as exc:
        logger.warning("Could not write workflow metrics report: %s", exc)


def _maybe_append_dataset_example(
    *,
    initial_state_dict: dict,
    spreadsheet_path: str,
    final_state: dict,
    source_run_id: Optional[str],
) -> None:
    """
    Create a dataset example from the compact workflow I/O.

    Requires a LangSmith dataset of type **Key-value** (not Chat-only / LLM-only).
    Set either LANGSMITH_DATASET_ID (UUID from the dataset URL or settings) or
    LANGSMITH_DATASET_NAME (exact name in the UI). ID is preferred if names might collide.
    """
    dataset_id = (os.getenv("LANGSMITH_DATASET_ID") or "").strip()
    dataset_name = (os.getenv("LANGSMITH_DATASET_NAME") or "").strip()
    if not dataset_id and not dataset_name:
        return

    try:
        from langsmith import Client

        client = Client()
    except Exception as exc:  # pragma: no cover - import / auth edge cases
        logger.warning("LangSmith dataset sync skipped (client): %s", exc)
        return

    inputs = _compact_trace_inputs(
        {"initial_state_dict": initial_state_dict, "spreadsheet_path": spreadsheet_path}
    )
    outputs = _compact_trace_outputs(final_state)
    split = (os.getenv("LANGSMITH_DATASET_SPLIT") or "").strip() or None
    meta = {
        "workflow": "campaign_brief",
        "spreadsheet_path": spreadsheet_path,
    }
    try:
        kwargs: Dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "metadata": meta,
        }
        if split is not None:
            kwargs["split"] = split
        if source_run_id:
            kwargs["source_run_id"] = source_run_id
        if dataset_id:
            kwargs["dataset_id"] = dataset_id
        else:
            kwargs["dataset_name"] = dataset_name
        client.create_example(**kwargs)
    except Exception as exc:
        logger.warning("LangSmith dataset example not created: %s", exc)


@traceable(
    name="campaign_brief_workflow",
    run_type="chain",
    tags=["brief-tutor", "campaign_brief", "langgraph"],
    process_inputs=_compact_trace_inputs,
    process_outputs=_compact_trace_outputs,
)
def run_campaign_brief_workflow_traced(initial_state_dict: dict, spreadsheet_path: str) -> dict:
    """
    Run the compiled campaign workflow with LangGraph Runnable config for LangSmith
    and a root LangSmith run showing compact inputs/outputs.
    """
    root = get_current_run_tree()
    source_run_id = str(root.id) if root is not None else None

    workflow = create_campaign_workflow()
    start = time.perf_counter()
    final_state = workflow.invoke(
        initial_state_dict, config=_workflow_run_config(spreadsheet_path)
    )
    workflow_latency_ms = int((time.perf_counter() - start) * 1000)
    _append_workflow_metrics(spreadsheet_path, final_state, workflow_latency_ms)
    _write_workflow_metrics_report(spreadsheet_path, final_state, workflow_latency_ms)
    for alert in _emit_regression_alerts(final_state):
        logger.warning("Regression alert: %s", alert)

    _maybe_append_dataset_example(
        initial_state_dict=initial_state_dict,
        spreadsheet_path=spreadsheet_path,
        final_state=final_state,
        source_run_id=source_run_id,
    )
    return final_state
