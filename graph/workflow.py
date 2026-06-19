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
from graph.models import AgentState, CampaignBrief, CampaignDiagnosis, Campaign, Assets, OfferDetails, StyleDescriptions
from graph.tools import get_available_tools
from graph.llm_provider import create_chat_model, get_primary_provider, get_fallback_provider
from graph.console_log import log_progress, log_analytics, log_verbose, show_console_analytics

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
    max_tokens = TASK_AGENT_MAX_TOKENS if node_name in ["theme_agent", "new_creative_agent", "campaign_update_agent", "document_creator_agent"] else DEFAULT_MAX_TOKENS
    
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
        # Vertex/Gemini requires non-empty "parts" in message turns, so we skip
        # empty text entries and guarantee at least one user turn exists.
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

                # Skip empty dict-based messages to avoid invalid Vertex payloads.
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
            return {
                "next": "final_results",
                "next_node": "diagnosis_formatter",
                "campaign_diagnoses": diagnoses,
                "rework_count": updated_rework_count,
                "metadata": {**(state.metadata or {}), "eval_metrics": eval_metrics}
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
    return {
        "next": "document_creator_agent",
        "next_node": "diagnosis_formatter",
        "diagnoses_json_path": str(output_path) if output_path else None,  # Pass path to JSON file
        "rework_count": state.rework_count or 0,
        "metadata": {**(state.metadata or {}), "eval_metrics": eval_metrics}
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
    
    campaign_brief = state.campaign_brief
    diagnoses = state.campaign_diagnoses or []
    
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
        "eval_metrics": (state.metadata or {}).get("eval_metrics", {}),
        "node_metrics": (state.metadata or {}).get("node_metrics", []),
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
        "messages": updated_messages
    }


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
        print("\n" + "="*80)
        print("📋 BRIEF CREATOR - FINDINGS")
        print("="*80)
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
                
                # Offer Details
                if campaign.offer_details:
                    offer = campaign.offer_details
                    print(f"   📌 Headline: {offer.headline[:75] if offer.headline else 'N/A'}")
                    print(f"   💰 Offer: {offer.offer[:100] if offer.offer else 'N/A'}")
                    print(f"   📄 Body: {offer.body[:100] + '...' if offer.body and len(offer.body) > 100 else (offer.body or 'N/A')}")
                    print(f"   🎯 CTA: {offer.cta[:30] if offer.cta else 'N/A'}")
                
                # Style Descriptions
                if campaign.style_descriptions:
                    style = campaign.style_descriptions
                    print(f"   🎨 Style Direction: {style.asset_style_direction[:50] + '...' if style.asset_style_direction and len(style.asset_style_direction) > 50 else (style.asset_style_direction or 'N/A')}")
                
                # Assets
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
        
        print("\n" + "="*80 + "\n")
    else:
        print("[Brief Creator] No campaign brief found")
    return result


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
    workflow.add_node("final_results", final_results_node)
    
    # Set entry point
    workflow.set_entry_point("brief_creator")
    
    # Add edges
    # Brief creator always goes to router
    workflow.add_edge("brief_creator", "router")
    
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
    
    # Final results node always ends the workflow
    workflow.add_edge("final_results", END)
    
    return workflow.compile()
