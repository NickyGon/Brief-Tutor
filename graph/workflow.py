"""
Main LangGraph workflow for the agent.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import yaml
import json
import re
import sys
import os
import time
import traceback
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from graph.brief_naming import parse_brief_filename, resolve_spreadsheet_path
from graph.models import AgentState, CampaignBrief, CampaignDiagnosis, Campaign, Assets, OfferDetails, StyleDescriptions
from graph.tools import (
    get_available_tools,
    extract_family_slug_from_filename,
    extract_campaign_instance_id,
    classify_campaign_path_hierarchy,
    list_same_family_local_spreadsheets,
    parse_local_spreadsheet_to_campaign_brief,
    compare_briefs_and_rank,
    resolve_global_campaign_pairs,
    write_family_similarity_outputs,
)
from graph.llm_provider import create_chat_model, get_primary_provider, get_fallback_provider
from graph.console_log import log_progress, log_analytics, log_verbose, show_console_analytics
from graph.supabase import is_supabase_configured, SupabaseWorkflowRepository

# Import from rag_ingestion (may need path adjustment)
try:
    import rag_ingestion
    store_diagnoses_to_drive = rag_ingestion.store_diagnoses_to_drive
except ImportError:
    # Will be imported dynamically if needed
    rag_ingestion = None
    store_diagnoses_to_drive = None



DEFAULT_MODEL = "gpt-5-nano" # Optimal for speed and cost
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7  # Optimal for RAG and structured data processing

# Higher token limits for agents that need to generate or evaluate multiple diagnoses
TASK_AGENT_MAX_TOKENS = 12000  # For theme_agent, new_creative_agent, campaign_update_agent, qa_agent

POST_PARSE_ROUTE_ENV = "BRIEF_POST_PARSE_ROUTE"


def _copy_metadata(metadata: Any) -> Dict[str, Any]:
    return metadata.copy() if isinstance(metadata, dict) else {}


def _get_db_context(metadata: Dict[str, Any]) -> Dict[str, Any]:
    db_context = metadata.get("db", {})
    if not isinstance(db_context, dict):
        db_context = {}
    warnings = db_context.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    db_context["warnings"] = warnings
    return db_context


def _append_db_warning(db_context: Dict[str, Any], message: str) -> None:
    warnings = db_context.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(message)
    db_context["warnings"] = warnings


def _persist_brief_and_campaigns(
    campaign_brief: CampaignBrief,
    metadata: Dict[str, Any],
    *,
    create_run_if_missing: bool = True,
) -> Dict[str, Any]:
    db_context = _get_db_context(metadata)
    db_context["enabled"] = bool(is_supabase_configured(use_service_role=True))
    metadata["db"] = db_context
    if not db_context["enabled"]:
        return metadata

    try:
        repo = SupabaseWorkflowRepository(use_service_role=True)
        brief_row = repo.upsert_brief(campaign_brief)
        brief_fk = brief_row.get("id")
        if isinstance(brief_fk, int):
            db_context["brief_fk"] = brief_fk
            repo.upsert_campaigns(brief_fk=brief_fk, campaign_brief=campaign_brief)
            campaign_id_map = repo.get_campaign_id_map(brief_fk)
            file_name = Path(campaign_brief.spreadsheet_path).name
            campaign_maps = db_context.get("campaign_ids_by_file", {})
            if not isinstance(campaign_maps, dict):
                campaign_maps = {}
            campaign_maps[file_name] = campaign_id_map
            db_context["campaign_ids_by_file"] = campaign_maps
            db_context["brief_id"] = brief_row.get("brief_id")

            if create_run_if_missing and not db_context.get("run_fk"):
                route_value = str(os.getenv(POST_PARSE_ROUTE_ENV, "1")).strip()
                route_type = "similarity_only" if route_value == "0" else "standard_analyzer"
                run_row = repo.create_analysis_run(
                    route_type=route_type,
                    brief_fk=brief_fk,
                    config_snapshot={
                        "brief_post_parse_route": route_value,
                        "family_sim_strong_threshold": float(os.getenv("FAMILY_SIM_STRONG_THRESHOLD", "0.80")),
                        "family_sim_review_threshold": float(os.getenv("FAMILY_SIM_REVIEW_THRESHOLD", "0.50")),
                    },
                )
                run_fk = run_row.get("id")
                if isinstance(run_fk, int):
                    db_context["run_fk"] = run_fk
                    db_context["run_uuid"] = run_row.get("run_uuid")
                    db_context["route_type"] = route_type
    except Exception as exc:
        _append_db_warning(db_context, f"Brief/campaign persistence failed: {exc}")

    metadata["db"] = db_context
    return metadata


def _normalize_diagnoses_list(diagnoses: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for diagnosis in diagnoses:
        if isinstance(diagnosis, CampaignDiagnosis):
            normalized.append(
                {
                    "campaign_id": diagnosis.campaign_id,
                    "status": diagnosis.status,
                    "diagnosis": diagnosis.diagnosis,
                    "issues": diagnosis.issues or [],
                    "recommendations": diagnosis.recommendations or [],
                    "grounding_evidence": diagnosis.grounding_evidence or [],
                }
            )
        elif isinstance(diagnosis, dict):
            normalized.append(diagnosis)
    return normalized


def _persist_diagnoses(
    *,
    campaign_brief: Optional[CampaignBrief],
    diagnoses: List[Any],
    eval_metrics: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    db_context = _get_db_context(metadata)
    if not db_context.get("enabled"):
        metadata["db"] = db_context
        return metadata

    run_fk = db_context.get("run_fk")
    brief_fk = db_context.get("brief_fk")
    if not isinstance(run_fk, int) or not isinstance(brief_fk, int):
        metadata["db"] = db_context
        return metadata

    normalized = _normalize_diagnoses_list(diagnoses)
    if not normalized:
        metadata["db"] = db_context
        return metadata

    try:
        repo = SupabaseWorkflowRepository(use_service_role=True)
        file_name = Path(campaign_brief.spreadsheet_path).name if campaign_brief else ""
        campaign_maps = db_context.get("campaign_ids_by_file", {})
        campaign_id_map = campaign_maps.get(file_name, {}) if isinstance(campaign_maps, dict) else {}
        if not campaign_id_map:
            campaign_id_map = repo.get_campaign_id_map(brief_fk)
            if isinstance(campaign_maps, dict) and file_name:
                campaign_maps[file_name] = campaign_id_map
                db_context["campaign_ids_by_file"] = campaign_maps

        diagnosis_rows = repo.build_diagnosis_rows(
            run_fk=run_fk,
            brief_fk=brief_fk,
            diagnoses=normalized,
            campaign_id_map=campaign_id_map,
            eval_metrics=eval_metrics,
        )
        inserted = repo.upsert_diagnosis_rows(diagnosis_rows)
        db_context["diagnoses_persisted"] = len(inserted)
    except Exception as exc:
        _append_db_warning(db_context, f"Diagnosis persistence failed: {exc}")

    metadata["db"] = db_context
    return metadata


def _persist_similarity_matches(
    campaign_brief: Optional[CampaignBrief],
    family_similarity_payload: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    db_context = _get_db_context(metadata)
    if not db_context.get("enabled"):
        metadata["db"] = db_context
        return metadata

    run_fk = db_context.get("run_fk")
    if not isinstance(run_fk, int):
        metadata["db"] = db_context
        return metadata

    strong_matches = family_similarity_payload.get("strong_matches", [])
    review_matches = family_similarity_payload.get("review_matches", [])
    absolute_pairs = family_similarity_payload.get("absolute_pairs", [])
    likely_pairs = family_similarity_payload.get("likely_pairs", [])
    review_pairs = family_similarity_payload.get("review_pairs", [])
    if not isinstance(strong_matches, list):
        strong_matches = []
    if not isinstance(review_matches, list):
        review_matches = []
    if not isinstance(absolute_pairs, list):
        absolute_pairs = []
    if not isinstance(likely_pairs, list):
        likely_pairs = []
    if not isinstance(review_pairs, list):
        review_pairs = []
    # Prefer explicit pairing lists when present.
    persist_strong = absolute_pairs + likely_pairs if (absolute_pairs or likely_pairs) else strong_matches
    persist_review = review_pairs if review_pairs else review_matches

    try:
        repo = SupabaseWorkflowRepository(use_service_role=True)
        campaign_maps = db_context.get("campaign_ids_by_file", {})
        if not isinstance(campaign_maps, dict):
            campaign_maps = {}

        target_file_name = family_similarity_payload.get("target_file_name") or (
            Path(campaign_brief.spreadsheet_path).name if campaign_brief else ""
        )
        needed_files = {str(target_file_name)}
        for match in strong_matches + review_matches:
            if isinstance(match, dict):
                needed_files.add(str(match.get("file_name", "")))

        for file_name in [name for name in needed_files if name]:
            if file_name in campaign_maps:
                continue
            brief_id = Path(file_name).stem
            brief_row = repo.get_brief_by_brief_id(brief_id)
            brief_fk = brief_row.get("id") if isinstance(brief_row, dict) else None
            if isinstance(brief_fk, int):
                campaign_maps[file_name] = repo.get_campaign_id_map(brief_fk)

        db_context["campaign_ids_by_file"] = campaign_maps
        lookup: Dict[tuple[str, str], int] = {}
        for file_name, map_data in campaign_maps.items():
            if not isinstance(map_data, dict):
                continue
            for campaign_ext, campaign_fk in map_data.items():
                if isinstance(campaign_fk, int):
                    lookup[(str(file_name), str(campaign_ext))] = campaign_fk

        strong_threshold = float(os.getenv("FAMILY_SIM_STRONG_THRESHOLD", "0.80"))
        review_threshold = float(os.getenv("FAMILY_SIM_REVIEW_THRESHOLD", "0.50"))
        if strong_threshold <= 1.0:
            strong_threshold *= 100.0
        if review_threshold <= 1.0:
            review_threshold *= 100.0

        match_rows = repo.build_similarity_rows(
            run_fk=run_fk,
            target_file_name=str(target_file_name),
            strong_matches=persist_strong,
            review_matches=persist_review,
            campaign_fk_lookup=lookup,
            strong_threshold=strong_threshold,
            review_threshold=review_threshold,
        )
        inserted = repo.upsert_similarity_rows(match_rows)
        db_context["similarity_matches_persisted"] = len(inserted)
    except Exception as exc:
        _append_db_warning(db_context, f"Similarity persistence failed: {exc}")

    metadata["db"] = db_context
    return metadata


def _finalize_run_if_needed(metadata: Dict[str, Any], status: str = "completed") -> Dict[str, Any]:
    db_context = _get_db_context(metadata)
    if not db_context.get("enabled"):
        metadata["db"] = db_context
        return metadata
    if db_context.get("run_finalized"):
        metadata["db"] = db_context
        return metadata

    run_fk = db_context.get("run_fk")
    if not isinstance(run_fk, int):
        metadata["db"] = db_context
        return metadata

    try:
        repo = SupabaseWorkflowRepository(use_service_role=True)
        repo.finish_analysis_run(run_fk, status=status)
        db_context["run_finalized"] = True
    except Exception as exc:
        _append_db_warning(db_context, f"Run finalization failed: {exc}")
    metadata["db"] = db_context
    return metadata


def load_agent_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load agent configuration from a YAML file.
    
    Args:
        yaml_path: Path to the agent YAML file
        
    Returns:
        Dictionary containing the agent configuration
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_agent_from_config(config_path: str, tools: List) -> Any:
    """
    Create a LangGraph agent from a YAML configuration file.
    
    Args:
        config_path: Path to the agent YAML configuration file
        tools: List of tools to bind to the agent
        node_name: Name of the node (for logging purposes)
        
    Returns:
        A LangGraph agent node function
    """
    config = load_agent_config(config_path)
    
    # Use config name for logging
    node_name = config.get("name", "agent").lower().replace(" ", "_")
    
    # Create LLM with specified parameters
    # Task-specific agents need more tokens to generate/validate diagnoses for multiple campaigns
    # Document creator agent also needs more tokens to generate two full documents
    max_tokens = TASK_AGENT_MAX_TOKENS if node_name in [
        "theme_agent",
        "new_creative_agent",
        "campaign_update_agent",
        "document_creator_agent",
        "family_similarity_agent",
    ] else DEFAULT_MAX_TOKENS
    
    primary_provider = get_primary_provider()
    fallback_provider = get_fallback_provider()
    llm = create_chat_model(
        provider=primary_provider,
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=max_tokens,
    )
    
    # Get the system prompt from config
    system_prompt = config.get("prompt", "")
    
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """
        Agent node that processes messages using the configured agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with agent response
        """
        messages = state.messages.copy() if state.messages else []
        
        # Add system message if prompt exists
        # Remove any existing system messages first to ensure each agent gets its own
        if system_prompt:
            # Remove all existing system messages (both dict format and SystemMessage objects)
            filtered_messages = []
            removed_count = 0
            for msg in messages:
                is_system = False
                if isinstance(msg, dict):
                    is_system = msg.get("role") == "system"
                elif isinstance(msg, SystemMessage):
                    is_system = True
                
                if not is_system:
                    filtered_messages.append(msg)
                else:
                    removed_count += 1
            
            if removed_count > 0:
                print(f"[{node_name}] Removed {removed_count} existing system message(s), adding agent-specific system message")
            
            messages = filtered_messages
            
            # Add the current agent's system message at the beginning
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # For task-specific agents (not brief_creator), add campaign_brief to initial message if available
        # This prevents them from reloading the spreadsheet during rework cycles
        if node_name in ["theme_agent", "new_creative_agent", "campaign_update_agent"]:
            if state.campaign_brief and not any(
                isinstance(msg, dict) and "campaign_brief" in str(msg.get("content", "")).lower() 
                for msg in messages
            ):
                brief_info = {
                    "spreadsheet_path": state.campaign_brief.spreadsheet_path,
                    "task_type": state.campaign_brief.task_type,
                    "asset_summary": state.campaign_brief.asset_summary,
                    "dealership_name": state.campaign_brief.dealership_name,
                    "content_11_20": state.campaign_brief.content_11_20,
                    "campaigns": [camp.model_dump(mode='python') for camp in state.campaign_brief.campaigns]
                }
                campaign_brief_message = {
                    "role": "user",
                    "content": f"The campaign brief is already loaded in the workflow state. Use this campaign brief data - DO NOT call load_and_parse_spreadsheet:\n\n{json.dumps(brief_info, indent=2)}"
                }
                # Insert after system message but before other messages
                insert_idx = 1 if messages and messages[0].get("role") == "system" else 0
                messages.insert(insert_idx, campaign_brief_message)
                
        
        # Convert dict messages to LangChain message objects.
        # Skip empty text entries and guarantee at least one user turn exists.
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)
                content = content.strip()

                # Skip empty dict-based messages to avoid invalid model payloads.
                if not content:
                    continue

                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            else:
                # Keep non-dict messages only when they carry non-empty content.
                msg_content = getattr(msg, "content", None)
                if msg_content is None:
                    continue
                if not isinstance(msg_content, str):
                    msg_content = str(msg_content)
                if not msg_content.strip():
                    continue
                langchain_messages.append(msg)

        # Ensure at least one user message exists for model input.
        if not any(isinstance(m, HumanMessage) for m in langchain_messages):
            langchain_messages.append(
                HumanMessage(
                    content="Proceed with the campaign analysis using the provided campaign brief and instructions."
                )
            )
        
        # Create agent with tools
        agent = create_agent(llm, tools)

        # Invoke agent with provider fallback and per-node metrics
        response = None
        fallback_used = False
        invoke_error = None
        start_time = time.perf_counter()
        try:
            response = agent.invoke({"messages": langchain_messages})
        except Exception as exc:
            invoke_error = exc
            if fallback_provider and fallback_provider != primary_provider:
                print(
                    f"[{node_name}] Primary provider '{primary_provider}' failed; retrying with fallback '{fallback_provider}'. Error: {exc}"
                )
                fallback_llm = create_chat_model(
                    provider=fallback_provider,
                    model=os.getenv("FALLBACK_LLM_MODEL", DEFAULT_MODEL),
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=max_tokens,
                )
                fallback_agent = create_agent(fallback_llm, tools)
                response = fallback_agent.invoke({"messages": langchain_messages})
                fallback_used = True
            else:
                raise

        # Some providers can return a response with finish_reason=MALFORMED_FUNCTION_CALL
        # without raising an exception. In that case, retry once with a strict repair hint.
        def _has_malformed_function_call(resp: Dict[str, Any]) -> bool:
            if not resp or not resp.get("messages"):
                return False
            for m in resp.get("messages", []):
                if not isinstance(m, AIMessage):
                    continue
                metadata = getattr(m, "response_metadata", {}) or {}
                if str(metadata.get("finish_reason", "")).upper() == "MALFORMED_FUNCTION_CALL":
                    return True
            return False

        if _has_malformed_function_call(response):
            print(f"[{node_name}] Detected MALFORMED_FUNCTION_CALL. Retrying once with tool-call repair instructions.")
            repair_messages = list(langchain_messages)
            repair_messages.append(
                HumanMessage(
                    content=(
                        "Your previous tool call was malformed. Retry now.\n"
                        "Use valid JSON arguments matching the tool schema exactly.\n"
                        "Call one tool at a time, then continue."
                    )
                )
            )
            response = agent.invoke({"messages": repair_messages})
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log_verbose(f"[{node_name}] Agent response: {response}")
        
        # Convert response back to dict format and extract tool results
        updated_messages = messages.copy()
        updated_campaign_brief = state.campaign_brief
        updated_diagnoses = state.campaign_diagnoses
        
        if response.get("messages"):
            for msg in response["messages"]:
                if isinstance(msg, AIMessage):
                    # Try to extract campaign_diagnoses from agent's response
                    content = msg.content
                    
                    # Check for empty content (might indicate token limit was hit)
                    if not content or (isinstance(content, str) and not content.strip()):
                        # Check if token limit was hit
                        metadata = getattr(msg, 'response_metadata', {})
                        finish_reason = metadata.get('finish_reason', '')
                        if finish_reason == 'length':
                            print(f"[{node_name}] ⚠️  WARNING: Agent response was truncated due to token limit!")
                            print(f"           Consider increasing max_tokens or reducing campaign count.")
                            if node_name in ["theme_agent", "new_creative_agent", "campaign_update_agent"]:
                                print(f"           Current max_tokens: {max_tokens}")
                        elif finish_reason:
                            print(f"[{node_name}] ⚠️  Agent response finished with reason: {finish_reason}")
                        else:
                            print(f"[{node_name}] ⚠️  WARNING: Agent returned empty content")
                    
                    if isinstance(content, str) and content.strip():
                        try:
                            # Try to parse as JSON
                            parsed = json.loads(content)
                            if isinstance(parsed, dict) and "campaign_diagnoses" in parsed:
                                diagnoses_data = parsed["campaign_diagnoses"]
                                
                                # Convert dicts to CampaignDiagnosis objects
                                if isinstance(diagnoses_data, list):
                                    updated_diagnoses = []
                                    for diag_data in diagnoses_data:
                                        if isinstance(diag_data, dict):
                                            updated_diagnoses.append(CampaignDiagnosis(**diag_data))
                                        elif isinstance(diag_data, CampaignDiagnosis):
                                            updated_diagnoses.append(diag_data)
                                    if updated_diagnoses:
                                        print(f"[{node_name}] ✓ Successfully extracted {len(updated_diagnoses)} diagnoses from JSON")
                                        updated_diagnoses = updated_diagnoses
                        except (json.JSONDecodeError, TypeError):
                            # Try to extract JSON from text
                            json_match = re.search(r'\{.*"campaign_diagnoses".*\}', content, re.DOTALL)
                            if json_match:
                                try:
                                    parsed = json.loads(json_match.group())
                                    if isinstance(parsed, dict) and "campaign_diagnoses" in parsed:
                                        diagnoses_data = parsed["campaign_diagnoses"]
                                        if isinstance(diagnoses_data, list):
                                            updated_diagnoses = []
                                            for diag_data in diagnoses_data:
                                                if isinstance(diag_data, dict):
                                                    updated_diagnoses.append(CampaignDiagnosis(**diag_data))
                                                elif isinstance(diag_data, CampaignDiagnosis):
                                                    updated_diagnoses.append(diag_data)
                                            if updated_diagnoses:
                                                print(f"[{node_name}] ✓ Successfully extracted {len(updated_diagnoses)} diagnoses from embedded JSON")
                                except Exception as e:
                                    print(f"[{node_name}] ⚠️  Failed to parse JSON from content: {e}")
                                    # Log a snippet of the content for debugging
                                    if content:
                                        print(f"           Content preview: {content[:200]}...")
                    
                    updated_messages.append({
                        "role": "assistant",
                        "content": content if content else ""
                    })
                elif isinstance(msg, ToolMessage):
                    # Extract tool results - look for CampaignBrief from load_and_parse_spreadsheet
                    tool_result = msg.content
                    
                    # Check if this is from the spreadsheet loader tools only.
                    # Do not infer from generic payload content (e.g. any JSON containing task_type),
                    # otherwise non-loader tools can accidentally overwrite state.campaign_brief.
                    tool_name = getattr(msg, 'name', None) or getattr(msg, 'tool_call_id', None)
                    is_spreadsheet_tool = (
                        "load_and_parse_spreadsheet" in str(tool_name).lower()
                    )
                    
                    # Try to parse tool_result as JSON if it's a string
                    if isinstance(tool_result, str):
                        try:
                            tool_result = json.loads(tool_result)
                        except (json.JSONDecodeError, TypeError):
                            # If it's not JSON, check if it contains campaign brief data
                            if is_spreadsheet_tool:
                                log_verbose(f"[DEBUG] Tool result is string but not JSON: {tool_result[:200]}")
                            pass
                    
                    # Only spreadsheet loader results should update the main campaign_brief state.
                    if is_spreadsheet_tool:
                        # Check if this is already a CampaignBrief object
                        if isinstance(tool_result, CampaignBrief):
                            updated_campaign_brief = tool_result
                            log_verbose(f"[DEBUG] Tool result is already a CampaignBrief with {len(tool_result.campaigns)} campaigns")
                        # Check if this is a CampaignBrief result (has task_type and campaigns)
                        elif isinstance(tool_result, dict) and "task_type" in tool_result and "campaigns" in tool_result:
                            try:
                                log_verbose(f"[DEBUG] Found CampaignBrief in tool result. Task type: {tool_result.get('task_type')}, Campaigns: {len(tool_result.get('campaigns', []))}")
                                
                                # Convert campaigns from dicts back to Campaign objects if needed
                                campaigns_data = tool_result.get("campaigns", [])
                                # Reconstruct Campaign objects if they're dicts
                                reconstructed_campaigns = []
                                for camp_data in campaigns_data:
                                    if isinstance(camp_data, dict):
                                        try:
                                            reconstructed_campaigns.append(Campaign(**camp_data))
                                        except Exception as e:
                                            print(f"[ERROR] Failed to reconstruct Campaign from dict: {type(e).__name__}: {str(e)}")
                                            print(f"       Campaign data keys: {list(camp_data.keys()) if isinstance(camp_data, dict) else 'N/A'}")
                                            # Try to reconstruct nested objects
                                            try:
                                                # Handle nested Pydantic models
                                                if "assets" in camp_data and isinstance(camp_data["assets"], dict):
                                                    camp_data["assets"] = Assets(**camp_data["assets"])
                                                if "offer_details" in camp_data and isinstance(camp_data["offer_details"], dict):
                                                    camp_data["offer_details"] = OfferDetails(**camp_data["offer_details"])
                                                if "style_descriptions" in camp_data and isinstance(camp_data["style_descriptions"], dict):
                                                    camp_data["style_descriptions"] = StyleDescriptions(**camp_data["style_descriptions"])
                                                reconstructed_campaigns.append(Campaign(**camp_data))
                                            except Exception as e2:
                                                print(f"[ERROR] Failed to reconstruct Campaign even with nested objects: {type(e2).__name__}: {str(e2)}")
                                    elif isinstance(camp_data, Campaign):
                                        reconstructed_campaigns.append(camp_data)
                                    else:
                                        print(f"[WARN] Unexpected campaign data type: {type(camp_data)}")
                                
                                tool_result["campaigns"] = reconstructed_campaigns
                                updated_campaign_brief = CampaignBrief(**tool_result)
                                log_verbose(f"[DEBUG] Successfully created CampaignBrief with {len(updated_campaign_brief.campaigns)} campaigns")
                            except Exception as e:
                                # If parsing fails, log the error for debugging
                                print(f"[ERROR] Failed to create CampaignBrief from tool result: {type(e).__name__}: {str(e)}")
                                print(f"       Tool result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'N/A'}")
                                traceback.print_exc()
                    
                    # Ensure tool_result is JSON serializable
                    def make_json_serializable(obj):
                        """Recursively convert Pydantic models and other non-serializable objects to dicts/strings"""
                        if hasattr(obj, 'model_dump'):
                            # Pydantic model
                            return obj.model_dump(mode='python')
                        elif isinstance(obj, dict):
                            return {k: make_json_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [make_json_serializable(item) for item in obj]
                        elif isinstance(obj, (str, int, float, bool, type(None))):
                            return obj
                        else:
                            # Fallback to string representation
                            return str(obj)
                    
                    try:
                        serializable_result = make_json_serializable(tool_result)
                        tool_content = json.dumps(serializable_result) if isinstance(serializable_result, dict) else str(serializable_result)
                    except Exception as e:
                        print(f"[WARN] Failed to serialize tool result to JSON: {type(e).__name__}: {str(e)}")
                        tool_content = str(tool_result)
                    
                    updated_messages.append({
                        "role": "tool",
                        "name": str(tool_name) if tool_name else "",
                        "tool_call_id": str(getattr(msg, "tool_call_id", "") or ""),
                        "content": tool_content
                    })
        
        state_metadata = state.metadata.copy() if isinstance(state.metadata, dict) else {}
        node_metrics = state_metadata.get("node_metrics", [])
        if not isinstance(node_metrics, list):
            node_metrics = []
        node_metrics.append(
            {
                "node": node_name,
                "provider": fallback_provider if fallback_used else primary_provider,
                "primary_provider": primary_provider,
                "fallback_provider": fallback_provider,
                "fallback_used": fallback_used,
                "latency_ms": elapsed_ms,
                "error": str(invoke_error)[:300] if invoke_error else None,
            }
        )
        state_metadata["node_metrics"] = node_metrics

        result = {"messages": updated_messages, "next_node": node_name, "metadata": state_metadata}

        if updated_campaign_brief:
            result["campaign_brief"] = updated_campaign_brief
        if updated_diagnoses:
            result["campaign_diagnoses"] = updated_diagnoses
        
        # Print diagnoses for task-specific agents before going to QA
        if node_name in ["theme_agent", "new_creative_agent", "campaign_update_agent"]:
            agent_display_names = {
                "theme_agent": "THEME AGENT",
                "new_creative_agent": "NEW CREATIVE AGENT",
                "campaign_update_agent": "CAMPAIGN UPDATE AGENT"
            }
            agent_display_name = agent_display_names.get(node_name, node_name.upper())
            
            if updated_diagnoses and len(updated_diagnoses) > 0:
                log_progress(f"[{agent_display_name}] Produced {len(updated_diagnoses)} diagnosis(es).")
                status_counts = {"critical": 0, "observed": 0, "passed": 0}
                for diag in updated_diagnoses:
                    if isinstance(diag, CampaignDiagnosis):
                        status = diag.status
                    elif isinstance(diag, dict):
                        status = diag.get("status", "")
                    else:
                        continue
                    if status in status_counts:
                        status_counts[status] += 1
                if show_console_analytics():
                    log_analytics("\n" + "=" * 80)
                    log_analytics(f"🔍 {agent_display_name} - DIAGNOSES RESULTS")
                    log_analytics("=" * 80)
                    log_analytics(f"\n📊 Total Diagnoses: {len(updated_diagnoses)}")
                    log_analytics(f"\n📈 Status Breakdown:")
                    log_analytics(f"   🔴 Critical: {status_counts['critical']}")
                    log_analytics(f"   🟡 Observed: {status_counts['observed']}")
                    log_analytics(f"   🟢 Passed: {status_counts['passed']}")
                    log_analytics(f"\n📝 Detailed Diagnoses:")
                    log_analytics("-" * 80)
                    for i, diag in enumerate(updated_diagnoses, 1):
                        if isinstance(diag, CampaignDiagnosis):
                            campaign_id = diag.campaign_id
                            status = diag.status
                            diagnosis = diag.diagnosis
                            issues = diag.issues
                            recommendations = diag.recommendations
                        elif isinstance(diag, dict):
                            campaign_id = diag.get("campaign_id", "Unknown")
                            status = diag.get("status", "unknown")
                            diagnosis = diag.get("diagnosis", "")
                            issues = diag.get("issues", [])
                            recommendations = diag.get("recommendations", [])
                        else:
                            continue
                        status_emoji = {"critical": "🔴", "observed": "🟡", "passed": "🟢"}.get(status, "⚪")
                        log_analytics(
                            f"\n{i}. {status_emoji} Campaign: {campaign_id} [{status.upper()}]"
                        )
                        log_analytics(
                            f"   Diagnosis: {diagnosis[:200] + '...' if len(diagnosis) > 200 else diagnosis}"
                        )
                        if issues:
                            log_analytics(f"   Issues ({len(issues)}):")
                            for issue in issues[:3]:
                                log_analytics(f"     - {issue[:100] + '...' if len(issue) > 100 else issue}")
                            if len(issues) > 3:
                                log_analytics(f"     ... and {len(issues) - 3} more issues")
                        if recommendations:
                            log_analytics(f"   Recommendations ({len(recommendations)}):")
                            for rec in recommendations[:3]:
                                log_analytics(f"     - {rec[:100] + '...' if len(rec) > 100 else rec}")
                            if len(recommendations) > 3:
                                log_analytics(
                                    f"     ... and {len(recommendations) - 3} more recommendations"
                                )
                    log_analytics("\n" + "=" * 80 + "\n")
            else:
                log_progress(f"\n[WARN] {agent_display_name} did not produce any diagnoses.")
        
        # Preserve existing state fields
        if state.rework_count is not None:
            result["rework_count"] = state.rework_count
        if state.qa_result is not None:
            result["qa_result"] = state.qa_result
        if state.qa_feedback:
            result["qa_feedback"] = state.qa_feedback
        
        return result
    
    return agent_node


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router node that determines which agent to use based on task_type from campaign_brief.
    
    Routing logic:
    - "Theme" → theme_agent
    - "New Creative" → new_creative_agent
    - "Campaign Update" → campaign_update_agent
    - Default (or unknown) → new_creative_agent
    
    Args:
        state: Current agent state (should contain campaign_brief with task_type)
        
    Returns:
        Updated state with:
        - next: The next node to route to (based on campaign_brief.task_type)
        - next_node: "router" (for logging)
    """
    campaign_brief = state.campaign_brief
    task_type = None
    
    # Primary source: get task_type from campaign_brief
    if campaign_brief and campaign_brief.task_type:
        task_type = campaign_brief.task_type
        print(f"[Router] Using task_type from campaign_brief: '{task_type}'")
    else:
        # Fallback: try to extract from messages (tool results)
        if state.messages:
            for msg in reversed(state.messages):
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    # Look for task_type in tool call results
                    if "task_type" in str(content).lower():
                        try:
                            if isinstance(content, str):
                                parsed = json.loads(content)
                                if isinstance(parsed, dict) and "task_type" in parsed:
                                    task_type = parsed["task_type"]
                                    print(f"[Router] Extracted task_type from messages: '{task_type}'")
                                    break
                        except (json.JSONDecodeError, TypeError):
                            pass
    
    # Default task_type if not found
    if not task_type:
        task_type = ""
        print("[Router] No task_type found, using default routing")
    
    task_type_lower = task_type.lower().strip()
    
    # Route based on task_type from campaign_brief
    if task_type_lower == "theme":
        next_node = "theme_agent"
    elif task_type_lower == "new creative":
        next_node = "new_creative_agent"
    elif task_type_lower == "campaign update":
        next_node = "campaign_update_agent"
    else:
        next_node = "new_creative_agent"  # Default fallback
    
    print(f"[Router] Routing to '{next_node}' based on task_type: '{task_type}'")
    
    return {
        "next": next_node,
        "next_node": "router"
    }


def diagnosis_formatter_node(state: AgentState) -> Dict[str, Any]:
    """
    Diagnosis formatter node that validates diagnoses, formats them to Excel, and keeps them in state.
    
    Steps:
    1. Structure validation (required fields, status values)
    2. Lightweight RAG validation (quick check against rules)
    3. Format to Excel using store_diagnoses_to_drive
    4. Keep diagnoses in state for Document Creator Agent
    5. Route based on validation result (max 3 rework attempts)
    
    Args:
        state: Current agent state with campaign diagnoses
        
    Returns:
        Updated state with formatted diagnoses and routing decision
    """
    diagnoses = state.campaign_diagnoses or []
    
    # Extract diagnoses from messages if not in state
    if not diagnoses and state.messages:
        for msg in reversed(state.messages):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "campaign_diagnoses" in parsed:
                            diagnoses_data = parsed["campaign_diagnoses"]
                            if isinstance(diagnoses_data, list):
                                diagnoses = []
                                for diag_data in diagnoses_data:
                                    if isinstance(diag_data, dict):
                                        diagnoses.append(CampaignDiagnosis(**diag_data))
                                    elif isinstance(diag_data, CampaignDiagnosis):
                                        diagnoses.append(diag_data)
                                break
                    except (json.JSONDecodeError, TypeError):
                        json_match = re.search(r'\{.*"campaign_diagnoses".*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group())
                                diagnoses_data = parsed.get("campaign_diagnoses")
                                if isinstance(diagnoses_data, list):
                                    diagnoses = []
                                    for diag_data in diagnoses_data:
                                        if isinstance(diag_data, dict):
                                            diagnoses.append(CampaignDiagnosis(**diag_data))
                                        elif isinstance(diag_data, CampaignDiagnosis):
                                            diagnoses.append(diag_data)
                                    break
                            except:
                                pass
    
    if not diagnoses:
        print("[Diagnosis Formatter] ⚠️  No diagnoses found in state")
        return {
            "next": "final_results",
            "next_node": "diagnosis_formatter",
            "campaign_diagnoses": []
        }
    
    min_groundedness = float(os.getenv("EVAL_MIN_GROUNDEDNESS", "0.8"))
    min_field_completeness = float(os.getenv("EVAL_MIN_FIELD_COMPLETENESS", "1.0"))

    # 1. Structure validation
    validation_errors = []
    grounded_count = 0
    complete_fields_count = 0
    for i, diag in enumerate(diagnoses):
        if isinstance(diag, dict):
            campaign_id = diag.get("campaign_id", "")
            status = diag.get("status", "")
            diagnosis = diag.get("diagnosis", "")
            issues = diag.get("issues", [])
            recommendations = diag.get("recommendations", [])
            grounding_evidence = diag.get("grounding_evidence", [])
        elif isinstance(diag, CampaignDiagnosis):
            campaign_id = diag.campaign_id
            status = diag.status
            diagnosis = diag.diagnosis
            issues = diag.issues or []
            recommendations = diag.recommendations or []
            grounding_evidence = diag.grounding_evidence or []
        else:
            validation_errors.append(f"Diagnosis {i+1}: Invalid format")
            continue
        
        # Check required fields
        if not campaign_id:
            validation_errors.append(f"Diagnosis {i+1}: Missing campaign_id")
        if not diagnosis:
            validation_errors.append(f"Diagnosis {i+1}: Missing diagnosis")
        if not status:
            validation_errors.append(f"Diagnosis {i+1}: Missing status")
        elif status not in ["critical", "observed", "passed"]:
            validation_errors.append(f"Diagnosis {i+1}: Invalid status '{status}' (must be 'critical', 'observed', or 'passed')")
        
        # Validate issues and recommendations are lists
        if not isinstance(issues, list):
            validation_errors.append(f"Diagnosis {i+1}: Issues must be a list")
        if not isinstance(recommendations, list):
            validation_errors.append(f"Diagnosis {i+1}: Recommendations must be a list")
        if not isinstance(grounding_evidence, list):
            validation_errors.append(f"Diagnosis {i+1}: grounding_evidence must be a list")

        has_required_fields = bool(campaign_id and status and diagnosis and isinstance(issues, list) and isinstance(recommendations, list))
        if has_required_fields:
            complete_fields_count += 1
        if isinstance(grounding_evidence, list) and any(str(item).strip() for item in grounding_evidence):
            grounded_count += 1
    
    total_diagnoses = len(diagnoses)
    groundedness_score = (grounded_count / total_diagnoses) if total_diagnoses else 0.0
    field_completeness_score = (complete_fields_count / total_diagnoses) if total_diagnoses else 0.0

    if groundedness_score < min_groundedness:
        validation_errors.append(
            f"Groundedness score {groundedness_score:.2f} is below threshold {min_groundedness:.2f}. "
            "Each diagnosis should include grounding_evidence entries."
        )
    if field_completeness_score < min_field_completeness:
        validation_errors.append(
            f"Field completeness score {field_completeness_score:.2f} is below threshold {min_field_completeness:.2f}."
        )

    # 2. Lightweight eval gate
    validation_passed = len(validation_errors) == 0
    eval_metrics = {
        "grounded_count": grounded_count,
        "total_diagnoses": total_diagnoses,
        "groundedness_score": round(groundedness_score, 4),
        "field_completeness_score": round(field_completeness_score, 4),
        "min_groundedness": min_groundedness,
        "min_field_completeness": min_field_completeness,
        "passed": validation_passed,
    }
    
    if not validation_passed:
        print(f"[Diagnosis Formatter] ⚠️  Validation failed with {len(validation_errors)} errors:")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        
        # Route back to task agent for rework
        updated_rework_count = (state.rework_count or 0) + 1
        
        if updated_rework_count >= 3:
            print(f"[Diagnosis Formatter] Max reworks ({updated_rework_count}) reached. Proceeding to final_results.")
            state_metadata = _copy_metadata(state.metadata)
            state_metadata["eval_metrics"] = eval_metrics
            state_metadata = _persist_diagnoses(
                campaign_brief=state.campaign_brief,
                diagnoses=diagnoses,
                eval_metrics=eval_metrics,
                metadata=state_metadata,
            )
            return {
                "next": "final_results",
                "next_node": "diagnosis_formatter",
                "campaign_diagnoses": diagnoses,
                "rework_count": updated_rework_count,
                "metadata": state_metadata,
            }
        
        # Determine which agent to route back to
        task_type_agent = "new_creative_agent"  # default
        if state.campaign_brief and state.campaign_brief.task_type:
            task_type_lower = state.campaign_brief.task_type.lower().strip()
            if task_type_lower == "theme":
                task_type_agent = "theme_agent"
            elif task_type_lower == "new creative":
                task_type_agent = "new_creative_agent"
            elif task_type_lower == "campaign update":
                task_type_agent = "campaign_update_agent"
        
        print(f"[Diagnosis Formatter] Validation failed (attempt {updated_rework_count}/3). Routing back to {task_type_agent} for rework.")
        
        # Add feedback message
        updated_messages = state.messages.copy() if state.messages else []
        feedback_text = "Validation failed. Issues found:\n" + "\n".join(validation_errors[:10])
        updated_messages.append({
            "role": "system",
            "content": f"⚠️ REWORK MODE (Attempt {updated_rework_count}/3) ⚠️\n\nDiagnosis Formatter Validation Feedback:\n{feedback_text}\n\nPlease fix the validation errors and resubmit your diagnoses."
        })
        
        return {
            "next": task_type_agent,
            "next_node": "diagnosis_formatter",
            "campaign_diagnoses": diagnoses,
            "rework_count": updated_rework_count,
            "messages": updated_messages,
            "metadata": {**(state.metadata or {}), "eval_metrics": eval_metrics}
        }
    
    # 3. Save diagnoses locally as JSON file
    print(f"[Diagnosis Formatter] ✓ Validation passed. Saving {len(diagnoses)} diagnoses locally as JSON...")
    
    try:
        # Extract filename from spreadsheet_path
        spreadsheet_path = state.campaign_brief.spreadsheet_path if state.campaign_brief else ""
        filename_base = ""
        
        if spreadsheet_path:
            # Extract filename from path (handle both local paths and URLs)
            if "/" in spreadsheet_path:
                filename_base = spreadsheet_path.split("/")[-1]
            elif "\\" in spreadsheet_path:
                filename_base = spreadsheet_path.split("\\")[-1]
            else:
                filename_base = spreadsheet_path
            
            # Remove extension if present
            if "." in filename_base:
                filename_base = filename_base.rsplit(".", 1)[0]
        
        if not filename_base:
            # Fallback: use date-based filename
            current_date = datetime.now()
            date_str = current_date.strftime("%Y-%m-%d")
            filename_base = f"{date_str}-diagnoses"
        
        # Create output filename: [filename]-diagnoses.json
        output_filename = f"{filename_base}-diagnoses.json"
        
        # Convert diagnoses to JSON-serializable format
        diagnoses_list = []
        for diag in diagnoses:
            if isinstance(diag, CampaignDiagnosis):
                diagnoses_list.append({
                    "campaign_id": diag.campaign_id,
                    "status": diag.status,
                    "diagnosis": diag.diagnosis,
                    "issues": diag.issues or [],
                    "recommendations": diag.recommendations or [],
                    "grounding_evidence": diag.grounding_evidence or [],
                })
            elif isinstance(diag, dict):
                diagnoses_list.append(diag)
        
        # Prepare output data
        output_data = {
            "task_type": state.campaign_brief.task_type if state.campaign_brief else None,
            "dealership_name": state.campaign_brief.dealership_name if state.campaign_brief else None,
            "asset_summary": state.campaign_brief.asset_summary if state.campaign_brief else None,
            "spreadsheet_path": spreadsheet_path,
            "diagnosis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_campaigns": len(diagnoses_list),
            "campaign_diagnoses": diagnoses_list,
            "eval_metrics": eval_metrics,
        }
        
        # Save to local file
        project_root = Path(__file__).parent.parent
        output_path = project_root / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Diagnosis Formatter] ✓ Diagnoses saved locally: {output_path}")
        print(f"     File: {output_filename}")
        
        log_analytics(f"\n[Diagnosis Formatter] 📄 Diagnoses JSON Content:")
        log_analytics("-" * 80)
        log_analytics(json.dumps(output_data, indent=2, ensure_ascii=False))
        log_analytics("-" * 80)
            
    except Exception as e:
        # Log error but don't fail the workflow if storage fails
        print(f"[Diagnosis Formatter] Warning: Failed to save diagnoses locally: {e}")
        traceback.print_exc()
        output_path = None
        output_filename = None
    
    # Store the JSON file path in state for Document Creator Agent
    # Route to document_creator_agent (which will read from the JSON file and create text documents)
    if output_path:
        print(f"\n[Diagnosis Formatter] ✓ JSON file created. Proceeding to document_creator_agent to create text documents.")
    else:
        print(f"\n[Diagnosis Formatter] ⚠️  JSON file creation failed. Proceeding to document_creator_agent anyway (it will handle the error).")
    state_metadata = _copy_metadata(state.metadata)
    state_metadata["eval_metrics"] = eval_metrics
    state_metadata = _persist_diagnoses(
        campaign_brief=state.campaign_brief,
        diagnoses=diagnoses,
        eval_metrics=eval_metrics,
        metadata=state_metadata,
    )
    return {
        "next": "document_creator_agent",
        "next_node": "diagnosis_formatter",
        "diagnoses_json_path": str(output_path) if output_path else None,  # Pass path to JSON file
        "rework_count": state.rework_count or 0,
        "metadata": state_metadata,
    }


def document_creator_node(state: AgentState) -> Dict[str, Any]:
    """
    Document creator node that reads diagnoses from JSON file, invokes the document creator agent
    to generate brief resume and full diagnoses listing documents, then saves them locally.
    
    Args:
        state: Current agent state with diagnoses_json_path
        
    Returns:
        Updated state with document metadata and routing to final_results
    """
    print(f"\n[Document Creator] Starting document creation process...")
    
    # Get the JSON file path from state
    diagnoses_json_path = getattr(state, 'diagnoses_json_path', None)
    if not diagnoses_json_path:
        # Try to get from state dict if it's a dict
        if isinstance(state, dict):
            diagnoses_json_path = state.get('diagnoses_json_path')
        else:
            diagnoses_json_path = None
    
    if not diagnoses_json_path or not Path(diagnoses_json_path).exists():
        print(f"[Document Creator] ⚠️  ERROR: Diagnoses JSON file not found!")
        print(f"     Expected path: {diagnoses_json_path}")
        print(f"     Cannot create documents without diagnosis data.")
        return {
            "next": "final_results",
            "next_node": "document_creator_agent",
            "messages": state.messages if hasattr(state, 'messages') else []
        }
    
    print(f"[Document Creator] 📖 Reading diagnoses from JSON file: {diagnoses_json_path}")
    
    # Read the diagnoses JSON file
    try:
        with open(diagnoses_json_path, 'r', encoding='utf-8') as f:
            diagnoses_json = json.load(f)
        
        diagnoses_data = diagnoses_json.get("campaign_diagnoses", [])
        task_type = diagnoses_json.get("task_type")
        dealership_name = diagnoses_json.get("dealership_name")
        spreadsheet_path = diagnoses_json.get("spreadsheet_path")
        
        print(f"[Document Creator] ✓ Loaded {len(diagnoses_data)} diagnoses from JSON file")
        print(f"     Task Type: {task_type}")
        print(f"     Dealership: {dealership_name}")
        
    except Exception as e:
        print(f"[Document Creator] ⚠️  ERROR: Failed to read diagnoses JSON file: {e}")
        traceback.print_exc()
        return {
            "next": "final_results",
            "next_node": "document_creator_agent",
            "messages": state.messages if hasattr(state, 'messages') else []
        }
    
    # Get the document creator agent
    document_creator_agent = create_agent_from_config(
        "agents/document_creator_agent.yaml",
        get_available_tools("document_creator_agent")
    )
    
    # Prepare state for the agent
    agent_state = state.model_copy()
    if agent_state.messages is None:
        agent_state.messages = []
    
    # Count campaigns by status for reference
    critical_count = sum(1 for d in diagnoses_data if d.get("status") == "critical")
    observed_count = sum(1 for d in diagnoses_data if d.get("status") == "observed")
    passed_count = sum(1 for d in diagnoses_data if d.get("status") == "passed")
    total_count = len(diagnoses_data)
    
    log_analytics(
        f"[Document Creator] 📊 Campaign counts: Critical={critical_count}, "
        f"Observed={observed_count}, Passed={passed_count}, Total={total_count}"
    )
    
    # Extract filename base from the JSON file path (which was created by diagnosis formatter)
    filename_base = ""
    if diagnoses_json_path:
        json_filename = Path(diagnoses_json_path).name
        # Remove "-diagnoses.json" suffix to get base filename
        if json_filename.endswith("-diagnoses.json"):
            filename_base = json_filename[:-len("-diagnoses.json")]
        elif json_filename.endswith(".json"):
            filename_base = json_filename[:-5]
        else:
            filename_base = json_filename
    
    if not filename_base:
        # Fallback: use spreadsheet path from JSON
        if spreadsheet_path:
            if "/" in spreadsheet_path:
                filename_base = spreadsheet_path.split("/")[-1]
            elif "\\" in spreadsheet_path:
                filename_base = spreadsheet_path.split("\\")[-1]
            else:
                filename_base = spreadsheet_path
            
            if "." in filename_base:
                filename_base = filename_base.rsplit(".", 1)[0]
    
    if not filename_base:
        current_date = datetime.now()
        date_str = current_date.strftime("%Y-%m-%d")
        filename_base = f"{date_str}-diagnoses"
    
    print(f"[Document Creator] 📝 Using filename base: {filename_base}")
    
    diagnoses_message = {
        "role": "user",
        "content": f"""You must create the Brief Resume and Full Diagnoses Listing documents using the ACTUAL diagnosis data below.

IMPORTANT: Use ONLY the data provided below. Do NOT use example or template text.

Campaign Diagnoses Data (use this exact data):
{json.dumps(diagnoses_data, indent=2)}

Campaign Brief Metadata:
- Task Type: {task_type or 'N/A'}
- Dealership Name: {dealership_name or 'N/A'}
- Spreadsheet Path: {spreadsheet_path or 'N/A'}

Reference counts (verify these match the data above):
- Critical: {critical_count}
- Observed: {observed_count}
- Passed: {passed_count}
- Total: {total_count}

CRITICAL INSTRUCTIONS:
1. For the Brief Resume:
   - Use the template format: 📊 Campaign Brief Diagnosis Summary, then separator lines, then emoji stats
   - Fill in the numbers: 🔴 {critical_count} / {total_count}, 🟡 {observed_count} / {total_count}, 🟢 {passed_count} / {total_count}
   - List each campaign from the "Campaign Diagnoses Data" above using its ACTUAL campaign_id field
   - For each campaign, use the ACTUAL issues array and recommendations array from that diagnosis object
   - Do NOT use example campaign IDs like "12345" or "67890" - use the real campaign_id values from the data

2. For the Full Listing:
   - List ALL campaigns from the "Campaign Diagnoses Data" above
   - For each campaign, include: campaign_id, status, diagnosis, issues, recommendations from the data
   - Use the actual values from the JSON data provided

3. Use the `write_document_to_file` tool to save both documents:
   - Filename base: {filename_base}
   - Call the tool TWICE:
     * First call: file_name="{filename_base}-resume.txt", document_type="brief_resume", content=[your generated brief resume text]
     * Second call: file_name="{filename_base}-details.txt", document_type="full_listing", content=[your generated full listing text]
   - Generate the complete document content before calling the tool
   - Use the ACTUAL data from the Campaign Diagnoses Data above

DO NOT copy example text. Generate everything from the actual diagnosis data provided above."""
    }
    
    # Add the message to the state
    agent_state.messages = agent_state.messages + [diagnoses_message]
    
    # Verify tools are available
    tools_list = get_available_tools("document_creator_agent")
    tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools_list]
    print(f"[Document Creator] 🔧 Available tools: {tool_names}")
    
    # Invoke the agent
    print(f"[Document Creator] 🤖 Invoking document creator agent to generate and write documents...")
    result = document_creator_agent(agent_state)
    print(f"[Document Creator] ✓ Agent response received")
    
    log_verbose("[Document Creator] 🔍 Debugging agent response...")
    if result.get("messages"):
        log_verbose(f"     Total messages in response: {len(result['messages'])}")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__ if not isinstance(msg, dict) else msg.get("role", "unknown")
            log_verbose(f"     Message {i}: {msg_type}")
            if isinstance(msg, dict):
                log_verbose(f"       Keys: {list(msg.keys())}")
                if "tool_calls" in msg:
                    log_verbose(f"       Tool calls: {msg['tool_calls']}")
                if "content" in msg:
                    content_preview = str(msg["content"])[:200] if msg["content"] else "None"
                    log_verbose(f"       Content preview: {content_preview}...")
            elif hasattr(msg, 'tool_calls'):
                log_verbose(f"       Tool calls attribute: {msg.tool_calls}")
            elif hasattr(msg, 'content'):
                content_preview = str(msg.content)[:200] if msg.content else "None"
                log_verbose(f"       Content preview: {content_preview}...")
    
    # Check for tool calls in the agent response
    # The agent should have called write_document_to_file twice (once for brief_resume, once for full_listing)
    tool_calls_found = []
    brief_resume_written = False
    full_listing_written = False
    
    if result.get("messages"):
        for msg in result["messages"]:
            # Handle both dict and LangChain message objects
            tool_calls = None
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls", [])
                role = msg.get("role", "")
            else:
                # LangChain message object
                if hasattr(msg, 'tool_calls'):
                    tool_calls = msg.tool_calls or []
                role = getattr(msg, 'role', '')
            
            # Check for tool calls
            if tool_calls:
                log_verbose(f"[Document Creator] 🔍 Found {len(tool_calls)} tool call(s)")
                for tool_call in tool_calls:
                    # Handle both dict and LangChain tool call formats
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        args = tool_call.get("args", {})
                    else:
                        tool_name = getattr(tool_call, 'name', '')
                        args = getattr(tool_call, 'args', {}) or {}
                    
                    log_verbose(f"     Tool call: {tool_name} with args: {args}")
                    if tool_name == "write_document_to_file":
                        tool_calls_found.append(tool_call)
                        doc_type = args.get("document_type", "") if isinstance(args, dict) else getattr(args, 'document_type', '')
                        file_name = args.get("file_name", "") if isinstance(args, dict) else getattr(args, 'file_name', '')
                        if doc_type == "brief_resume":
                            brief_resume_written = True
                            print(f"[Document Creator] ✓ Tool call detected: Brief Resume -> {file_name}")
                        elif doc_type == "full_listing":
                            full_listing_written = True
                            print(f"[Document Creator] ✓ Tool call detected: Full Listing -> {file_name}")
            
            # Also check tool message responses (tool results)
            if role == "tool" or (isinstance(msg, dict) and msg.get("role") == "tool"):
                tool_name = ""
                content = ""
                if isinstance(msg, dict):
                    tool_name = msg.get("name", "")
                    content = msg.get("content", "")
                else:
                    tool_name = getattr(msg, 'name', '')
                    content = getattr(msg, 'content', '')
                
                # Some wrappers may omit tool name in serialized messages; in that case,
                # detect document writes from the tool result payload itself.
                is_write_doc_tool = "write_document_to_file" in str(tool_name).lower()
                try:
                    if isinstance(content, str):
                        tool_result = json.loads(content)
                    else:
                        tool_result = content

                    if isinstance(tool_result, dict) and tool_result.get("success"):
                        doc_type = str(tool_result.get("document_type", "")).strip().lower()
                        file_name = tool_result.get("file_name", "")
                        if doc_type == "brief_resume":
                            brief_resume_written = True
                            print(f"[Document Creator] ✓ Tool result: Brief Resume written successfully -> {file_name}")
                        elif doc_type == "full_listing":
                            full_listing_written = True
                            print(f"[Document Creator] ✓ Tool result: Full Listing written successfully -> {file_name}")
                        elif is_write_doc_tool:
                            # Tool name matches but doc_type was unexpected.
                            print(f"[Document Creator] ⚠️  Unexpected document_type in tool result: {tool_result.get('document_type')}")
                except Exception as e:
                    # Only warn when this looked like the write tool, otherwise ignore
                    # unrelated tool payloads that are not JSON.
                    if is_write_doc_tool:
                        print(f"[Document Creator] ⚠️  Error parsing tool result: {e}")
                        print(f"     Tool result content: {str(content)[:200]}")
    
    # Verify that both documents were written via tool calls
    print(f"\n[Document Creator] 📋 Checking tool call results...")
    
    if brief_resume_written and full_listing_written:
        print(f"[Document Creator] ✓ Both documents were written successfully via tool calls")
        print(f"     - Brief Resume: Written via write_document_to_file tool")
        print(f"     - Full Listing: Written via write_document_to_file tool")
    else:
        if not brief_resume_written:
            print(f"[Document Creator] ⚠️  WARNING: Brief Resume was not written via tool call")
            print(f"     The agent should have called write_document_to_file with document_type='brief_resume'")
        if not full_listing_written:
            print(f"[Document Creator] ⚠️  WARNING: Full Listing was not written via tool call")
            print(f"     The agent should have called write_document_to_file with document_type='full_listing'")
        
        if tool_calls_found:
            print(f"[Document Creator] Found {len(tool_calls_found)} tool call(s), but not all documents were written")
        else:
            print(f"[Document Creator] ⚠️  ERROR: No tool calls found in agent response")
            print(f"     The agent should use the write_document_to_file tool to save both documents")
            print(f"     Check the agent's response messages for errors or missing tool calls")
    
    print(f"\n[Document Creator] ✓ Document creation process complete. Proceeding to final_results.")
    
    return {
        "next": "final_results",
        "next_node": "document_creator_agent",
        "campaign_diagnoses": state.campaign_diagnoses,  # Keep diagnoses
        "messages": result.get("messages", state.messages)
    }


def final_results_node(state: AgentState) -> Dict[str, Any]:
    """
    Final results node that formats and delivers the workflow results in a friendly, structured format.
    
    Args:
        state: Current agent state with campaign brief and diagnoses
        
    Returns:
        Updated state with final_results containing structured output
    """
    
    state_metadata = _copy_metadata(state.metadata)
    state_metadata = _finalize_run_if_needed(state_metadata, status="completed")
    campaign_brief = state.campaign_brief
    family_similarity = state.family_similarity or {}
    diagnoses = state.campaign_diagnoses or []

    # Similarity-only route output.
    if family_similarity:
        final_results = {
            "mode": "family_similarity",
            "task_type": campaign_brief.task_type if campaign_brief else family_similarity.get("task_type"),
            "total_campaigns": len(campaign_brief.campaigns) if campaign_brief else 0,
            "family_similarity": family_similarity,
            "node_metrics": state_metadata.get("node_metrics", []),
            "db": state_metadata.get("db", {}),
            "route_toggle_value": str(os.getenv(POST_PARSE_ROUTE_ENV, "1")).strip(),
        }
        friendly_message = (
            "Completed family similarity analysis.\n"
            f"Target file: {family_similarity.get('target_file_name', 'N/A')}\n"
            f"Absolute pairs: {len(family_similarity.get('absolute_pairs', family_similarity.get('strong_matches', [])))}\n"
            f"Likely pairs: {len(family_similarity.get('likely_pairs', []))}\n"
            f"Review pairs: {len(family_similarity.get('review_pairs', family_similarity.get('review_matches', [])))}\n"
            f"Unpaired targets: {len(family_similarity.get('unpaired_targets', []))}.\n"
            "See final_results for structured output."
        )
        updated_messages = state.messages.copy() if state.messages else []
        updated_messages.append({"role": "assistant", "content": friendly_message})
        print(
            "[Final Results] Similarity route completed. "
            f"Absolute={len(family_similarity.get('absolute_pairs', family_similarity.get('strong_matches', [])))}, "
            f"Likely={len(family_similarity.get('likely_pairs', []))}, "
            f"Review={len(family_similarity.get('review_pairs', family_similarity.get('review_matches', [])))}"
        )
        return {
            "final_results": final_results,
            "next_node": "final_results",
            "messages": updated_messages,
            "metadata": state_metadata,
        }
    
    # Build the structured response
    task_type = campaign_brief.task_type if campaign_brief else "Unknown"
    total_campaigns = len(campaign_brief.campaigns) if campaign_brief else 0
    
    # Convert diagnoses to dict format for JSON serialization
    diagnoses_list = []
    for diag in diagnoses:
        if isinstance(diag, CampaignDiagnosis):
            diagnoses_list.append({
                "campaign_id": diag.campaign_id,
                "status": diag.status,
                "diagnosis": diag.diagnosis,
                "issues": diag.issues,
                "recommendations": diag.recommendations,
                "grounding_evidence": diag.grounding_evidence,
            })
        elif isinstance(diag, dict):
            diagnoses_list.append(diag)
    
    # Count statuses for additional context
    status_counts = {"critical": 0, "observed": 0, "passed": 0}
    for diag in diagnoses:
        if isinstance(diag, CampaignDiagnosis):
            status = diag.status
        elif isinstance(diag, dict):
            status = diag.get("status", "")
        else:
            continue
        if status in status_counts:
            status_counts[status] += 1
    
    # Create structured final results (JSON-like format for frontend)
    final_results = {
        "task_type": task_type,
        "total_campaigns": total_campaigns,
        "campaign_diagnoses": diagnoses_list,
        "status_breakdown": {
            "critical": status_counts["critical"],
            "observed": status_counts["observed"],
            "passed": status_counts["passed"]
        },
        "eval_metrics": state_metadata.get("eval_metrics", {}),
        "node_metrics": state_metadata.get("node_metrics", []),
        "db": state_metadata.get("db", {}),
        "qa_result": state.qa_result,
        "rework_count": state.rework_count or 0
    }
    
    # Friendly message for UI/consumers (metrics stay in final_results, not echoed here)
    friendly_message = (
        f"Completed analysis of your Campaign Brief.\n"
        f"Task type: {task_type}\n"
        f"Campaigns: {total_campaigns}\n"
        f"Diagnoses: {len(diagnoses_list)} "
        f"({status_counts['critical']} critical, {status_counts['observed']} observed, "
        f"{status_counts['passed']} passed).\n"
        f"See final_results for structured output."
    )
    
    # Add final message to state
    updated_messages = state.messages.copy() if state.messages else []
    updated_messages.append({
        "role": "assistant",
        "content": friendly_message
    })
    
    print(f"[Final Results] Workflow completed. Task Type: {task_type}, Campaigns: {total_campaigns}, Diagnoses: {len(diagnoses_list)}")
    
    return {
        "final_results": final_results,
        "next_node": "final_results",
        "messages": updated_messages,
        "metadata": state_metadata,
    }


def _print_campaign_brief_findings(
    campaign_brief: CampaignBrief,
    *,
    title: str = "BRIEF CREATOR - FINDINGS",
) -> None:
    """Print brief findings in the same style used by brief_creator."""
    print("\n" + "=" * 80)
    print(f"📋 {title}")
    print("=" * 80)
    print(f"\n📁 Spreadsheet Path: {campaign_brief.spreadsheet_path}")
    print(f"📌 Task Type: {campaign_brief.task_type}")
    print(f"🏢 Dealership Name: {campaign_brief.dealership_name or 'N/A'}")
    print(f"📊 Asset Summary: {campaign_brief.asset_summary or 'N/A'}")
    print(f"📑 Content 11-20: {'Yes' if campaign_brief.content_11_20 else 'No'}")
    print(f"\n📈 Total Campaigns: {len(campaign_brief.campaigns)}")

    if campaign_brief.campaigns:
        print("\n📝 Campaigns Found:")
        print("-" * 80)
        for i, campaign in enumerate(campaign_brief.campaigns, 1):
            print(f"\n{i}. Campaign ID: {campaign.campaign_id}")

            if campaign.offer_details:
                offer = campaign.offer_details
                print(f"   📌 Headline: {offer.headline[:75] if offer.headline else 'N/A'}")
                print(f"   💰 Offer: {offer.offer[:100] if offer.offer else 'N/A'}")
                print(
                    f"   📄 Body: {offer.body[:100] + '...' if offer.body and len(offer.body) > 100 else (offer.body or 'N/A')}"
                )
                print(f"   🎯 CTA: {offer.cta[:30] if offer.cta else 'N/A'}")

            if campaign.style_descriptions:
                style = campaign.style_descriptions
                print(
                    f"   🎨 Style Direction: {style.asset_style_direction[:80] + '...' if style.asset_style_direction and len(style.asset_style_direction) > 80 else (style.asset_style_direction or 'N/A')}"
                )
                if style.additional_style_information:
                    extra = style.additional_style_information
                    print(
                        f"   🧾 Additional Style: {extra[:80] + '...' if len(extra) > 80 else extra}"
                    )
                if style.vehicle_photography:
                    print(f"   📷 Vehicle Photography: {style.vehicle_photography}")
                if style.logos:
                    print(f"   🏷️  Logos: {style.logos}")

            if campaign.assets:
                assets = campaign.assets
                asset_codes = []
                if assets.sl_bn_srp_da:
                    asset_codes.append("SL/BN/SRP/DA")
                if assets.sl_m_bn_m:
                    asset_codes.append("SL_M/BN_M")
                if assets.facebook_assets:
                    asset_codes.append("Facebook")
                if assets.instagram_assets:
                    asset_codes.append("Instagram")
                if assets.google_assets:
                    asset_codes.append("Google")
                if assets.ot_1:
                    asset_codes.append("OT1")
                if assets.ot_2:
                    asset_codes.append("OT2")
                if assets.ot_3:
                    asset_codes.append("OT3")
                if assets.ot_4:
                    asset_codes.append("OT4")
                if assets.ot_5:
                    asset_codes.append("OT5")
                if assets.ot_6:
                    asset_codes.append("OT6")
                print(f"   🖼️  Assets: {', '.join(asset_codes) if asset_codes else 'None'}")

    print("\n" + "=" * 80 + "\n")


def brief_creator_node(state: AgentState) -> Dict[str, Any]:
    """
    Brief creator node wrapper that prints findings to console.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with campaign_brief
    """
    # Get the actual brief creator agent
    brief_creator_agent = create_agent_from_config(
        "agents/brief_creator_agent.yaml", 
        get_available_tools("brief_creator")
    )
    
    # Invoke the agent
    result = brief_creator_agent(state)

    log_verbose(f"[Brief Creator] Result: {result}")
    # Print findings to console if campaign_brief was extracted
    if "campaign_brief" in result and result["campaign_brief"]:
        campaign_brief = result["campaign_brief"]
        state_metadata = _copy_metadata(result.get("metadata") or state.metadata)
        state_metadata = _persist_brief_and_campaigns(
            campaign_brief=campaign_brief,
            metadata=state_metadata,
            create_run_if_missing=True,
        )
        result["metadata"] = state_metadata
        _print_campaign_brief_findings(campaign_brief, title="BRIEF CREATOR - FINDINGS")
    else:
        print("[Brief Creator] No campaign brief found")
    return result


def _process_similarity_candidates(
    *,
    target_brief: CampaignBrief,
    candidate_paths: List[str],
    strong_threshold: float,
    review_threshold: float,
    state_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_files: List[Dict[str, Any]] = []
    collected_pairs: List[Dict[str, Any]] = []
    warnings: List[str] = []
    updated_metadata = state_metadata

    for idx, candidate_path in enumerate(candidate_paths, 1):
        try:
            print(
                f"\n[Family Similarity] Parsing candidate {idx}/{len(candidate_paths)}: "
                f"{Path(candidate_path).name}"
            )
            candidate_brief = parse_local_spreadsheet_to_campaign_brief(candidate_path)
            _print_campaign_brief_findings(
                candidate_brief,
                title=(
                    f"FAMILY SIMILARITY - CANDIDATE {idx}/{len(candidate_paths)} FINDINGS "
                    f"({Path(candidate_path).name})"
                ),
            )
            updated_metadata = _persist_brief_and_campaigns(
                campaign_brief=candidate_brief,
                metadata=updated_metadata,
                create_run_if_missing=False,
            )
            comparison = compare_briefs_and_rank(
                target_brief=target_brief,
                candidate_brief=candidate_brief,
                strong_threshold=strong_threshold,
                review_threshold=review_threshold,
            )
            candidate_entry = {
                "file_name": Path(candidate_path).name,
                "file_path": candidate_path,
                "dealership_name": candidate_brief.dealership_name,
                "task_type": candidate_brief.task_type,
                "file_similarity_score": comparison.get("file_similarity_score", 0.0),
                "component_averages": comparison.get("component_averages", {}),
                "absolute_pairs": comparison.get("absolute_pairs", [])[:5],
                "likely_pairs": comparison.get("likely_pairs", [])[:5],
                "strongest_matches": comparison.get("best_campaign_matches", [])[:5],
            }
            candidate_files.append(candidate_entry)
            components = comparison.get("component_averages", {}) or {}
            print(
                f"[Family Similarity] Candidate score: "
                f"{float(candidate_entry['file_similarity_score']) * 100:.2f}% "
                f"(strongest scored) | campaigns={len(candidate_brief.campaigns)} | "
                f"absolute={len(comparison.get('absolute_pairs', []) or [])} "
                f"likely={len(comparison.get('likely_pairs', []) or [])} "
                f"review={len(comparison.get('review_pairs', []) or [])}"
            )
            print(
                "[Family Similarity] Breakdown: "
                f"styleDirection={float(components.get('style_direction_similarity', components.get('asset_and_style_similarity', 0.0))) * 100:.1f}% "
                f"(styleFields={float(components.get('style_fields_similarity', 0.0)) * 100:.1f}%, "
                f"assets={float(components.get('asset_structure_similarity', 0.0)) * 100:.1f}%) | "
                f"wording={float(components.get('campaign_wording_similarity', 0.0)) * 100:.1f}% | "
                f"dealership={float(components.get('dealership_relationship', 0.0)) * 100:.1f}% | "
                f"refStrength={float(components.get('reference_strength', components.get('reference_id_boost', 0.0))) * 100:.1f}%"
            )
            best_raw = comparison.get("best_raw_pair")
            if isinstance(best_raw, dict):
                print(
                    "[Family Similarity] Best raw pair: "
                    f"{best_raw.get('target_campaign_id')} <-> {best_raw.get('candidate_campaign_id')} | "
                    f"{float(best_raw.get('similarity_score', 0.0)) * 100:.2f}% | "
                    f"status={best_raw.get('pair_status', 'none')} | "
                    f"basis={best_raw.get('pair_basis', best_raw.get('scoring_path', 'n/a'))}"
                )

            seen_keys = set()
            for list_key in ("absolute_pairs", "likely_pairs", "review_pairs"):
                for match in comparison.get(list_key, []) or []:
                    if not isinstance(match, dict):
                        continue
                    enriched = {
                        **match,
                        "file_name": candidate_entry["file_name"],
                        "file_path": candidate_entry["file_path"],
                        "dealership_name": candidate_entry["dealership_name"],
                        "task_type": candidate_entry["task_type"],
                        "similarity_percent": round(
                            float(match.get("similarity_score", 0.0)) * 100, 2
                        ),
                    }
                    key = (
                        f"{enriched.get('target_campaign_id')}|"
                        f"{enriched.get('candidate_campaign_id')}|"
                        f"{enriched.get('file_name')}|"
                        f"{enriched.get('pair_status')}"
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    collected_pairs.append(enriched)
        except Exception as exc:
            warning = f"Skipped candidate '{candidate_path}': {exc}"
            warnings.append(warning)
            print(f"[Family Similarity] {warning}")

    candidate_files.sort(key=lambda item: float(item.get("file_similarity_score", 0.0)), reverse=True)
    return {
        "candidate_files": candidate_files,
        "collected_pairs": collected_pairs,
        "warnings": warnings,
        "metadata": updated_metadata,
    }


def _filter_candidates_with_oem_metadata(
    *,
    target_path: str,
    candidate_paths: List[str],
    state_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    db_context = _get_db_context(state_metadata)
    if not db_context.get("enabled"):
        return {"candidate_paths": candidate_paths, "warnings": []}

    try:
        target_meta = parse_brief_filename(target_path) or {}
        target_account = str(target_meta.get("account_id", "")).strip().lower()
        if not target_account:
            return {"candidate_paths": candidate_paths, "warnings": ["OEM filter skipped: could not parse target accountID."]}

        repo = SupabaseWorkflowRepository(use_service_role=True)
        source_metadata = repo.get_dealership_account_metadata(target_account)
        if not source_metadata:
            return {"candidate_paths": candidate_paths, "warnings": [f"OEM filter skipped: no metadata for account '{target_account}'."]}

        filtered: List[str] = []
        for candidate in candidate_paths:
            candidate_meta = parse_brief_filename(candidate) or {}
            candidate_account = str(candidate_meta.get("account_id", "")).strip().lower()
            if not candidate_account:
                continue
            candidate_account_meta = repo.get_dealership_account_metadata(candidate_account)
            if repo.oem_compatible(source_metadata, candidate_account_meta):
                filtered.append(candidate)

        return {"candidate_paths": filtered, "warnings": []}
    except Exception as exc:
        return {"candidate_paths": candidate_paths, "warnings": [f"OEM filter failed: {exc}"]}


def family_similarity_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Similarity-only branch node (route=0).
    Parses same-family spreadsheets and computes weighted campaign similarity.
    """
    campaign_brief = state.campaign_brief
    state_metadata = _copy_metadata(state.metadata)
    if not campaign_brief:
        payload = {
            "error": "campaign_brief not found in state",
            "target_file_name": "",
            "absolute_pairs": [],
            "likely_pairs": [],
            "review_pairs": [],
            "unpaired_targets": [],
            "strong_matches": [],
            "review_matches": [],
            "candidate_files": [],
            "warnings": ["Brief creator did not produce campaign_brief; similarity analysis skipped."],
        }
        return {
            "family_similarity": payload,
            "next": "family_similarity_report",
            "next_node": "family_similarity_analysis",
            "metadata": state_metadata,
        }

    project_root = Path(__file__).parent.parent
    target_path = campaign_brief.spreadsheet_path or ""
    resolved_target_path = str(
        resolve_spreadsheet_path(
            target_path,
            project_root=project_root,
            strict=False,
        )
    ) if target_path else ""
    target_file_name = Path(resolved_target_path).name if resolved_target_path else ""
    family_slug = extract_family_slug_from_filename(target_path) if target_path else None
    dealership_id = family_slug or "unknown"
    spreadsheet_instance_id = extract_campaign_instance_id(target_path) or "unknown"

    strong_threshold = float(os.getenv("FAMILY_SIM_STRONG_THRESHOLD", "0.80"))
    review_threshold = float(os.getenv("FAMILY_SIM_REVIEW_THRESHOLD", "0.50"))
    qualifying_threshold = float(
        os.getenv("FAMILY_SIM_QUALIFYING_THRESHOLD", str(strong_threshold))
    )
    widen_if_no_qualifying = (
        str(os.getenv("FAMILY_SIM_WIDEN_IF_NO_QUALIFYING", "true")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    use_oem_filter = (
        str(os.getenv("FAMILY_SIM_USE_OEM_FILTER", "true")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    oem_fallback_if_empty = (
        str(os.getenv("FAMILY_SIM_OEM_FALLBACK_IF_EMPTY", "true")).strip().lower()
        in {"1", "true", "yes", "on"}
    )

    hierarchy_context = classify_campaign_path_hierarchy(resolved_target_path or target_path)
    has_group_folder = bool(hierarchy_context.get("has_group_folder"))

    account_candidate_paths = (
        list_same_family_local_spreadsheets(resolved_target_path or target_path, search_scope="account")
        if (resolved_target_path or target_path)
        else []
    )
    print(
        "[Family Similarity] Starting analysis for "
        f"{target_file_name} | family={dealership_id} | account-scope candidates={len(account_candidate_paths)}"
    )

    candidate_files: List[Dict[str, Any]] = []
    collected_pairs: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if not family_slug:
        warnings.append(
            "Could not extract family slug from target filename. "
            "Expected pattern: YYYY-MM-<accountID>-<A-|D-><id>.xlsx"
        )

    # Ensure target brief/campaign rows are persisted and mapped.
    state_metadata = _persist_brief_and_campaigns(
        campaign_brief=campaign_brief,
        metadata=state_metadata,
        create_run_if_missing=True,
    )

    account_results = _process_similarity_candidates(
        target_brief=campaign_brief,
        candidate_paths=account_candidate_paths,
        strong_threshold=strong_threshold,
        review_threshold=review_threshold,
        state_metadata=state_metadata,
    )
    candidate_files.extend(account_results["candidate_files"])
    collected_pairs.extend(account_results.get("collected_pairs", []) or [])
    warnings.extend(account_results["warnings"])
    state_metadata = account_results["metadata"]

    account_resolved = resolve_global_campaign_pairs(
        collected_pairs,
        target_campaign_ids=[c.campaign_id for c in campaign_brief.campaigns],
    )
    account_qualifying = any(
        float(item.get("similarity_score", 0.0)) >= qualifying_threshold
        for item in (account_resolved.get("absolute_pairs", []) or [])
        + (account_resolved.get("likely_pairs", []) or [])
    )

    widened_scope = None
    widened_candidate_paths: List[str] = []
    if widen_if_no_qualifying and not account_qualifying:
        widened_scope = "group" if has_group_folder else "campaigns"
        widened_candidate_paths = list_same_family_local_spreadsheets(
            resolved_target_path or target_path,
            search_scope=widened_scope,
        )
        print(
            "[Family Similarity] No qualifying absolute/likely pairs in account scope. "
            f"Widening scope to '{widened_scope}' with {len(widened_candidate_paths)} candidates."
        )

        if use_oem_filter and widened_candidate_paths:
            oem_filter_result = _filter_candidates_with_oem_metadata(
                target_path=resolved_target_path or target_path,
                candidate_paths=widened_candidate_paths,
                state_metadata=state_metadata,
            )
            warnings.extend(oem_filter_result.get("warnings", []))
            filtered_paths = oem_filter_result.get("candidate_paths", widened_candidate_paths)
            if filtered_paths:
                widened_candidate_paths = filtered_paths
            elif oem_fallback_if_empty:
                warnings.append(
                    "OEM hard filter returned zero candidates; falling back to unfiltered widened scope."
                )
            else:
                widened_candidate_paths = []
                warnings.append(
                    "OEM hard filter returned zero candidates and fallback is disabled."
                )

        widened_results = _process_similarity_candidates(
            target_brief=campaign_brief,
            candidate_paths=widened_candidate_paths,
            strong_threshold=strong_threshold,
            review_threshold=review_threshold,
            state_metadata=state_metadata,
        )
        candidate_files.extend(widened_results["candidate_files"])
        collected_pairs.extend(widened_results.get("collected_pairs", []) or [])
        warnings.extend(widened_results["warnings"])
        state_metadata = widened_results["metadata"]

    candidate_files.sort(key=lambda item: float(item.get("file_similarity_score", 0.0)), reverse=True)
    resolved = resolve_global_campaign_pairs(
        collected_pairs,
        target_campaign_ids=[c.campaign_id for c in campaign_brief.campaigns],
    )
    absolute_pairs = resolved.get("absolute_pairs", []) or []
    likely_pairs = resolved.get("likely_pairs", []) or []
    review_pairs = resolved.get("review_pairs", []) or []
    unpaired_targets = resolved.get("unpaired_targets", []) or []
    strong_matches = resolved.get("strong_matches", []) or []
    review_matches = resolved.get("review_matches", []) or []

    print(
        "[Family Similarity] Pairing summary: "
        f"absolute={len(absolute_pairs)} likely={len(likely_pairs)} "
        f"review={len(review_pairs)} unpaired={len(unpaired_targets)}"
    )

    payload = {
        "target_file_name": target_file_name,
        "target_file_path": resolved_target_path or target_path,
        "dealership_name": campaign_brief.dealership_name,
        "dealership_family_id": dealership_id,
        "spreadsheet_instance_id": spreadsheet_instance_id,
        "task_type": campaign_brief.task_type,
        "thresholds": {
            "strong": strong_threshold,
            "review": review_threshold,
            "absolute": strong_threshold,
            "likely": 0.70,
        },
        "discovery_config": {
            "qualifying_threshold": qualifying_threshold,
            "widen_if_no_qualifying": widen_if_no_qualifying,
            "widened_scope": widened_scope,
            "use_oem_filter": use_oem_filter,
            "oem_fallback_if_empty": oem_fallback_if_empty,
        },
        "hierarchy_context": hierarchy_context,
        "candidate_files": candidate_files,
        "absolute_pairs": absolute_pairs,
        "likely_pairs": likely_pairs,
        "review_pairs": review_pairs,
        "unpaired_targets": unpaired_targets,
        "strong_matches": strong_matches,
        "review_matches": review_matches,
        "candidates_processed": len(candidate_files),
        "candidates_discovered": len(account_candidate_paths) + len(widened_candidate_paths),
        "warnings": warnings,
    }
    return {
        "family_similarity": payload,
        "next": "family_similarity_report",
        "next_node": "family_similarity_analysis",
        "metadata": state_metadata,
    }


def family_similarity_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Persist family-similarity JSON/TXT outputs and end similarity branch.
    """
    campaign_brief = state.campaign_brief
    payload = state.family_similarity.copy() if isinstance(state.family_similarity, dict) else {}
    state_metadata = _copy_metadata(state.metadata)
    if not campaign_brief:
        payload["warnings"] = payload.get("warnings", []) + [
            "Missing campaign_brief; report files were not generated."
        ]
        return {
            "family_similarity": payload,
            "next": "final_results",
            "next_node": "family_similarity_report",
            "metadata": state_metadata,
        }

    try:
        output_paths = write_family_similarity_outputs(campaign_brief, payload)
        payload.update(output_paths)
        print(
            "[Family Similarity] Outputs written: "
            f"json={output_paths.get('json_path')} report={output_paths.get('report_path')}"
        )
    except Exception as exc:
        warnings = payload.get("warnings", [])
        warnings.append(f"Failed to write similarity outputs: {exc}")
        payload["warnings"] = warnings
        print(f"[Family Similarity] Failed to write outputs: {exc}")

    state_metadata = _persist_similarity_matches(
        campaign_brief=campaign_brief,
        family_similarity_payload=payload,
        metadata=state_metadata,
    )

    return {
        "family_similarity": payload,
        "next": "final_results",
        "next_node": "family_similarity_report",
        "metadata": state_metadata,
    }


def _parse_json_object(content: str) -> Optional[Dict[str, Any]]:
    text = str(content or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return None
        try:
            parsed = json.loads(json_match.group())
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _compact_similarity_matches_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send specialists a smaller JSON payload for faster LLM turns."""
    def _compact_match(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target_campaign_id": item.get("target_campaign_id"),
            "candidate_campaign_id": item.get("candidate_campaign_id"),
            "file_name": item.get("file_name"),
            "similarity_score": item.get("similarity_score"),
            "similarity_percent": item.get("similarity_percent"),
            "scoring_path": item.get("scoring_path"),
            "pair_status": item.get("pair_status"),
            "pair_basis": item.get("pair_basis"),
            "style_direction_similarity": item.get("style_direction_similarity"),
            "campaign_wording_similarity": item.get("campaign_wording_similarity"),
            "dealership_relationship": item.get("dealership_relationship"),
            "reference_strength": item.get("reference_strength", item.get("reference_id_boost")),
            "reference_boost_reasons": item.get("reference_boost_reasons", []),
            "has_copy_refer_signal": item.get("has_copy_refer_signal", False),
        }

    return {
        "target_file_name": payload.get("target_file_name"),
        "dealership_name": payload.get("dealership_name"),
        "dealership_family_id": payload.get("dealership_family_id"),
        "task_type": payload.get("task_type"),
        "absolute_pairs": [
            _compact_match(item)
            for item in (payload.get("absolute_pairs", []) or [])
            if isinstance(item, dict)
        ],
        "likely_pairs": [
            _compact_match(item)
            for item in (payload.get("likely_pairs", []) or [])
            if isinstance(item, dict)
        ],
        "strong_matches": [
            _compact_match(item)
            for item in (payload.get("strong_matches", []) or [])
            if isinstance(item, dict)
        ],
        "review_matches": [
            _compact_match(item)
            for item in (
                payload.get("review_pairs", [])
                or payload.get("review_matches", [])
                or []
            )
            if isinstance(item, dict)
        ],
        "unpaired_targets": list(payload.get("unpaired_targets", []) or []),
    }


def _invoke_similarity_specialist(config_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Single-turn specialist LLM call returning parsed JSON (or empty dict on failure)."""
    config = load_agent_config(config_path)
    system_prompt = config.get("prompt", "")
    specialist_name = config.get("name", Path(config_path).stem)
    llm = create_chat_model(
        provider=get_primary_provider(),
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=min(TASK_AGENT_MAX_TOKENS, 4000),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Analyze these family-similarity matches as {specialist_name}. "
                "Return strict JSON only.\n\n"
                f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
            )
        ),
    ]
    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        parsed = _parse_json_object(str(content or ""))
        return parsed or {}
    except Exception as exc:
        print(f"[Family Similarity] Specialist '{specialist_name}' failed: {exc}")
        return {"error": str(exc)}


def _match_key(item: Dict[str, Any]) -> str:
    return (
        f"{item.get('target_campaign_id', '')}|"
        f"{item.get('candidate_campaign_id', '')}|"
        f"{item.get('file_name', '')}"
    )


def _merge_specialist_notes(
    matches: List[Dict[str, Any]],
    specialist_payloads: Dict[str, Dict[str, Any]],
    list_key: str,
) -> None:
    lookups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for specialist, payload in specialist_payloads.items():
        enriched = payload.get(list_key, []) if isinstance(payload, dict) else []
        lookup: Dict[str, Dict[str, Any]] = {}
        if isinstance(enriched, list):
            for item in enriched:
                if isinstance(item, dict):
                    lookup[_match_key(item)] = item
        lookups[specialist] = lookup

    for item in matches:
        if not isinstance(item, dict):
            continue
        key = _match_key(item)
        notes: List[str] = []
        evidence: List[str] = []
        for specialist in ("reference", "style", "wording"):
            candidate = lookups.get(specialist, {}).get(key, {})
            note = str(candidate.get("specialist_notes", "")).strip()
            if note:
                notes.append(f"{specialist}: {note}")
            points = candidate.get("evidence_points", []) if isinstance(candidate, dict) else []
            if isinstance(points, list):
                evidence.extend(str(v) for v in points if str(v).strip())
        if notes:
            item["specialist_notes"] = notes
        if evidence:
            # Temporary merge bag for synthesizer; final evidence comes from synthesizer.
            item["specialist_evidence_points"] = evidence[:8]


def family_similarity_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Parallel specialist enrichment + synthesizer for similarity matches.
    Specialists: reference, style/assets, wording. Main agent merges final narrative.
    """
    payload = state.family_similarity.copy() if isinstance(state.family_similarity, dict) else {}
    state_metadata = _copy_metadata(state.metadata)
    strong_matches = payload.get("strong_matches", []) or []
    review_matches = payload.get("review_matches", []) or []
    absolute_pairs = payload.get("absolute_pairs", []) or []
    likely_pairs = payload.get("likely_pairs", []) or []
    review_pairs = payload.get("review_pairs", []) or []

    if not strong_matches and not review_matches and not absolute_pairs and not likely_pairs and not review_pairs:
        return {
            "family_similarity": payload,
            "next": "family_similarity_report",
            "next_node": "family_similarity_agent",
            "metadata": state_metadata,
        }

    compact_payload = _compact_similarity_matches_payload(payload)
    specialist_jobs = [
        ("reference", "agents/family_sim_reference_agent.yaml"),
        ("style", "agents/family_sim_style_agent.yaml"),
        ("wording", "agents/family_sim_wording_agent.yaml"),
    ]
    specialist_results: Dict[str, Dict[str, Any]] = {}
    print(
        "[Family Similarity] Running specialists in order "
        "(reference -> style -> wording), then synthesizer..."
    )
    for name, path in specialist_jobs:
        print(f"[Family Similarity] Specialist '{name}' starting...")
        try:
            result_payload = _invoke_similarity_specialist(path, compact_payload) or {}
            specialist_results[name] = result_payload
            ok = "error" not in result_payload
            print(
                f"[Family Similarity] Specialist '{name}' "
                f"{'delivered' if ok else 'failed'}."
            )
        except Exception as exc:
            specialist_results[name] = {"error": str(exc)}
            print(f"[Family Similarity] Specialist '{name}' raised: {exc}")

    for list_key in ("absolute_pairs", "likely_pairs", "review_pairs", "strong_matches", "review_matches"):
        matches = payload.get(list_key, []) or []
        if isinstance(matches, list):
            _merge_specialist_notes(matches, specialist_results, list_key)
            payload[list_key] = matches

    payload["specialist_outputs"] = {
        name: {
            "ok": "error" not in (result or {}),
            "absolute_count": len((result or {}).get("absolute_pairs", []) or []),
            "likely_count": len((result or {}).get("likely_pairs", []) or []),
            "strong_count": len((result or {}).get("strong_matches", []) or []),
            "review_count": len(
                (result or {}).get("review_pairs", [])
                or (result or {}).get("review_matches", [])
                or []
            ),
            "error": (result or {}).get("error"),
        }
        for name, result in specialist_results.items()
    }
    print(
        "[Family Similarity] All specialists finished. "
        "Passing merged notes to main similarity synthesizer..."
    )

    synthesizer = create_agent_from_config(
        "agents/family_similarity_agent.yaml",
        get_available_tools("family_similarity_agent"),
    )
    agent_state = state.model_copy()
    if agent_state.messages is None:
        agent_state.messages = []

    synthesis_payload = {
        "target_file_name": payload.get("target_file_name"),
        "dealership_name": payload.get("dealership_name"),
        "dealership_family_id": payload.get("dealership_family_id"),
        "task_type": payload.get("task_type"),
        "absolute_pairs": payload.get("absolute_pairs", []),
        "likely_pairs": payload.get("likely_pairs", []),
        "review_pairs": payload.get("review_pairs", payload.get("review_matches", [])),
        "unpaired_targets": payload.get("unpaired_targets", []),
        "strong_matches": payload.get("strong_matches", []),
        "review_matches": payload.get("review_matches", []),
        "specialist_outputs": payload.get("specialist_outputs", {}),
    }
    agent_state.messages = agent_state.messages + [
        {
            "role": "user",
            "content": (
                "Synthesize specialist notes into final `match_reason` and `evidence_points` "
                "for every paired item in `absolute_pairs`, `likely_pairs`, `review_pairs` "
                "(and compatibility lists `strong_matches` / `review_matches`). "
                "Keep scores and pair_status unchanged. Return strict JSON only.\n\n"
                f"{json.dumps(synthesis_payload, indent=2, ensure_ascii=False)}"
            ),
        }
    ]

    result = synthesizer(agent_state)
    updated_messages = result.get("messages", state.messages)

    assistant_content = ""
    if isinstance(updated_messages, list):
        for msg in reversed(updated_messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_content = str(msg.get("content", "") or "").strip()
                if assistant_content:
                    break

    parsed_payload = _parse_json_object(assistant_content)
    if isinstance(parsed_payload, dict):
        for list_key in ("absolute_pairs", "likely_pairs", "review_pairs", "strong_matches", "review_matches"):
            existing = payload.get(list_key, []) or []
            enriched = parsed_payload.get(list_key, []) if isinstance(parsed_payload.get(list_key), list) else []
            enriched_lookup = {
                _match_key(item): item
                for item in enriched
                if isinstance(item, dict)
            }
            merged: List[Dict[str, Any]] = []
            for item in existing:
                if not isinstance(item, dict):
                    continue
                candidate = enriched_lookup.get(_match_key(item), {})
                merged_item = dict(item)
                reason = str(candidate.get("match_reason", "")).strip() if isinstance(candidate, dict) else ""
                evidence = candidate.get("evidence_points", []) if isinstance(candidate, dict) else []
                if reason:
                    merged_item["match_reason"] = reason
                if isinstance(evidence, list) and evidence:
                    merged_item["evidence_points"] = [str(v) for v in evidence if str(v).strip()][:4]
                merged_item.pop("specialist_evidence_points", None)
                merged.append(merged_item)
            payload[list_key] = merged
    else:
        warnings = payload.get("warnings", [])
        warnings.append("Family similarity synthesizer response was not valid JSON; keeping score-only matches.")
        payload["warnings"] = warnings

    return {
        "family_similarity": payload,
        "next": "family_similarity_report",
        "next_node": "family_similarity_agent",
        "messages": updated_messages,
        "metadata": state_metadata,
    }


def create_campaign_workflow() -> Any:
    """
    Creates the complete LangGraph workflow for campaign processing.
    
    Returns:
        A compiled LangGraph workflow
    """
    # Create agent nodes from YAML configs with agent-specific tools
    brief_creator = brief_creator_node
    theme_agent = create_agent_from_config(
        "agents/theme_agent.yaml", 
        get_available_tools("theme_agent")
    )
    new_creative_agent = create_agent_from_config(
        "agents/new_creative_agent.yaml", 
        get_available_tools("new_creative_agent")
    )
    campaign_update_agent = create_agent_from_config(
        "agents/campaign_update_agent.yaml", 
        get_available_tools("campaign_update_agent")
    )
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("brief_creator", brief_creator)
    workflow.add_node("router", router_node)
    workflow.add_node("theme_agent", theme_agent)
    workflow.add_node("new_creative_agent", new_creative_agent)
    workflow.add_node("campaign_update_agent", campaign_update_agent)
    workflow.add_node("diagnosis_formatter", diagnosis_formatter_node)
    workflow.add_node("document_creator_agent", document_creator_node)
    workflow.add_node("family_similarity_analysis", family_similarity_analysis_node)
    workflow.add_node("family_similarity_agent", family_similarity_agent_node)
    workflow.add_node("family_similarity_report", family_similarity_report_node)
    workflow.add_node("final_results", final_results_node)
    
    # Set entry point
    workflow.set_entry_point("brief_creator")
    
    # Add edges
    # Brief creator routes conditionally:
    # BRIEF_POST_PARSE_ROUTE=1 -> original analyzer route
    # BRIEF_POST_PARSE_ROUTE=0 -> similarity-only route
    def route_after_brief_creator(state: AgentState) -> str:
        route_value = str(os.getenv(POST_PARSE_ROUTE_ENV, "1")).strip()
        return "family_similarity_analysis" if route_value == "0" else "router"

    workflow.add_conditional_edges(
        "brief_creator",
        route_after_brief_creator,
        {
            "router": "router",
            "family_similarity_analysis": "family_similarity_analysis",
        },
    )
    
    # Router conditionally routes to one of the three agents
    def route_to_agent(state: AgentState) -> str:
        """
        Helper function to route based on state.next.
        Uses next_node from state for logging purposes.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the next node to route to
        """
        # Log current node (router) and next node for debugging
        next_node_name = state.next or "new_creative_agent"
        
        # The state.next_node should already be set to "router" by router_node
        # This function just returns the routing decision
        return next_node_name
    
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "theme_agent": "theme_agent",
            "new_creative_agent": "new_creative_agent",
            "campaign_update_agent": "campaign_update_agent"
        }
    )
    
    # All task type agents go to diagnosis_formatter after processing
    workflow.add_edge("theme_agent", "diagnosis_formatter")
    workflow.add_edge("new_creative_agent", "diagnosis_formatter")
    workflow.add_edge("campaign_update_agent", "diagnosis_formatter")
    
    # Diagnosis formatter routes based on validation result
    def route_from_diagnosis_formatter(state: AgentState) -> str:
        """
        Route from diagnosis formatter based on validation result and rework count.
        
        Returns:
            "document_creator_agent" if validation passed
            "final_results" if max reworks reached
            The task type agent name if rework is needed
        """
        if state.next == "document_creator_agent":
            return "document_creator_agent"
        elif state.next == "final_results":
            return "final_results"
        # Otherwise route back to the task type agent for rework
        return state.next or "new_creative_agent"
    
    workflow.add_conditional_edges(
        "diagnosis_formatter",
        route_from_diagnosis_formatter,
        {
            "document_creator_agent": "document_creator_agent",
            "final_results": "final_results",
            "theme_agent": "theme_agent",
            "new_creative_agent": "new_creative_agent",
            "campaign_update_agent": "campaign_update_agent"
        }
    )
    
    # Document creator agent always goes to final_results
    workflow.add_edge("document_creator_agent", "final_results")

    # Similarity-only branch ends after report generation.
    workflow.add_edge("family_similarity_analysis", "family_similarity_agent")
    workflow.add_edge("family_similarity_agent", "family_similarity_report")
    workflow.add_edge("family_similarity_report", "final_results")
    
    # Final results node always ends the workflow
    workflow.add_edge("final_results", END)
    
    return workflow.compile()
