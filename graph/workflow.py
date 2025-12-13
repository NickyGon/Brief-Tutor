"""
Main LangGraph workflow for the agent.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from graph.models import AgentState, CampaignBrief, CampaignDiagnosis, Campaign, Assets, OfferDetails, StyleDescriptions
from graph.tools import get_available_tools



DEFAULT_MODEL = "gpt-5-nano" # Optimal for speed and cost
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7  # Optimal for RAG and structured data processing

# Higher token limits for agents that need to generate multiple diagnoses
TASK_AGENT_MAX_TOKENS = 4000  # For theme_agent, new_creative_agent, campaign_update_agent


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
    # Task-specific agents need more tokens to generate diagnoses for multiple campaigns
    max_tokens = TASK_AGENT_MAX_TOKENS if node_name in ["theme_agent", "new_creative_agent", "campaign_update_agent"] else DEFAULT_MAX_TOKENS
    
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=max_tokens
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
                import json
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
                
        
        # Convert dict messages to LangChain message objects
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(msg)
        
        # Create agent with tools
        agent = create_agent(llm, tools)
        
        # Invoke agent
        response = agent.invoke({"messages": langchain_messages})
        print(f"[{node_name}] Agent response: {response}")
        
        # Convert response back to dict format and extract tool results
        updated_messages = messages.copy()
        updated_campaign_brief = state.campaign_brief
        updated_diagnoses = state.campaign_diagnoses
        
        import json
        import re
        
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
                    
                    # Check if this is from load_and_parse_spreadsheet tool
                    tool_name = getattr(msg, 'name', None) or getattr(msg, 'tool_call_id', None)
                    is_spreadsheet_tool = (
                        "load_and_parse_spreadsheet" in str(tool_name).lower() or
                        "spreadsheet" in str(tool_result).lower() or
                        (isinstance(tool_result, (dict, str)) and "task_type" in str(tool_result))
                    )
                    
                    # Try to parse tool_result as JSON if it's a string
                    if isinstance(tool_result, str):
                        try:
                            tool_result = json.loads(tool_result)
                        except (json.JSONDecodeError, TypeError):
                            # If it's not JSON, check if it contains campaign brief data
                            if is_spreadsheet_tool:
                                print(f"[DEBUG] Tool result is string but not JSON: {tool_result[:200]}")
                            pass
                    
                    # Check if this is already a CampaignBrief object
                    if isinstance(tool_result, CampaignBrief):
                        updated_campaign_brief = tool_result
                        print(f"[DEBUG] Tool result is already a CampaignBrief with {len(tool_result.campaigns)} campaigns")
                    # Check if this is a CampaignBrief result (has task_type and campaigns)
                    elif isinstance(tool_result, dict) and "task_type" in tool_result and "campaigns" in tool_result:
                        try:
                            print(f"[DEBUG] Found CampaignBrief in tool result. Task type: {tool_result.get('task_type')}, Campaigns: {len(tool_result.get('campaigns', []))}")
                            
                            # Convert campaigns from dicts back to Campaign objects if needed
                            campaigns_data = tool_result.get("campaigns", [])
                            # Reconstruct Campaign objects if they're dicts
                            from graph.models import Campaign
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
                            print(f"[DEBUG] Successfully created CampaignBrief with {len(updated_campaign_brief.campaigns)} campaigns")
                        except Exception as e:
                            # If parsing fails, log the error for debugging
                            print(f"[ERROR] Failed to create CampaignBrief from tool result: {type(e).__name__}: {str(e)}")
                            print(f"       Tool result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'N/A'}")
                            import traceback
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
                        "content": tool_content
                    })
        
        result = {"messages": updated_messages, "next_node": node_name}

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
                print("\n" + "="*80)
                print(f"🔍 {agent_display_name} - DIAGNOSES RESULTS")
                print("="*80)
                print(f"\n📊 Total Diagnoses: {len(updated_diagnoses)}")
                
                # Count statuses
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
                
                print(f"\n📈 Status Breakdown:")
                print(f"   🔴 Critical: {status_counts['critical']}")
                print(f"   🟡 Observed: {status_counts['observed']}")
                print(f"   🟢 Passed: {status_counts['passed']}")
                
                print(f"\n📝 Detailed Diagnoses:")
                print("-" * 80)
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
                    print(f"\n{i}. {status_emoji} Campaign: {campaign_id} [{status.upper()}]")
                    print(f"   Diagnosis: {diagnosis[:200] + '...' if len(diagnosis) > 200 else diagnosis}")
                    
                    if issues:
                        print(f"   Issues ({len(issues)}):")
                        for issue in issues[:3]:  # Show first 3 issues
                            print(f"     - {issue[:100] + '...' if len(issue) > 100 else issue}")
                        if len(issues) > 3:
                            print(f"     ... and {len(issues) - 3} more issues")
                    
                    if recommendations:
                        print(f"   Recommendations ({len(recommendations)}):")
                        for rec in recommendations[:3]:  # Show first 3 recommendations
                            print(f"     - {rec[:100] + '...' if len(rec) > 100 else rec}")
                        if len(recommendations) > 3:
                            print(f"     ... and {len(recommendations) - 3} more recommendations")
                
                print("\n" + "="*80)
                print("➡️  Proceeding to QA Agent for review...")
                print("="*80 + "\n")
            else:
                print(f"\n[WARN] {agent_display_name} did not produce any diagnoses.")
        
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
    - "Rework" or "Campaign Update" → campaign_update_agent
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
                        import json
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
    elif task_type_lower in ["rework", "campaign update"]:
        next_node = "campaign_update_agent"
    else:
        next_node = "new_creative_agent"  # Default fallback
    
    print(f"[Router] Routing to '{next_node}' based on task_type: '{task_type}'")
    
    return {
        "next": next_node,
        "next_node": "router"
    }


def qa_node(state: AgentState) -> Dict[str, Any]:
    """
    QA node that reviews campaign diagnoses from the task type agent.
    
    Steps:
    1. Extract campaign diagnoses from agent messages
    2. Use QA agent to review diagnoses against RAG rules
    3. Determine if rework is needed (max 3 attempts)
    
    Args:
        state: Current agent state with campaign diagnoses
        
    Returns:
        Updated state with QA result and routing decision
    """
    # Extract diagnoses from the last agent's output
    # The diagnoses should be in the messages from the task type agent
    import json
    import re
    
    diagnoses = state.campaign_diagnoses
    if not diagnoses and state.messages:
        # Try to extract from messages
        for msg in reversed(state.messages):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "campaign_diagnoses" in parsed:
                            diagnoses_data = parsed["campaign_diagnoses"]
                            # Convert dicts to CampaignDiagnosis objects
                            if isinstance(diagnoses_data, list):
                                diagnoses = []
                                for diag_data in diagnoses_data:
                                    if isinstance(diag_data, dict):
                                        diagnoses.append(CampaignDiagnosis(**diag_data))
                                    elif isinstance(diag_data, CampaignDiagnosis):
                                        diagnoses.append(diag_data)
                            break
                    except (json.JSONDecodeError, TypeError):
                        # Check if content contains campaign_diagnoses as text
                        if "campaign_diagnoses" in content.lower():
                            try:
                                # Try to extract JSON from text
                                json_match = re.search(r'\{.*"campaign_diagnoses".*\}', content, re.DOTALL)
                                if json_match:
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
    
    # Get the QA agent
    qa_agent = create_agent_from_config(
        "agents/qa_agent.yaml",
        get_available_tools("qa_agent")
    )
    
    # Prepare state for QA agent
    qa_state = state.model_copy()
    if diagnoses:
        qa_state.campaign_diagnoses = diagnoses
    
    # Invoke QA agent
    qa_result = qa_agent(qa_state)
    
    # Extract QA result from QA agent's response
    qa_passed = None
    qa_feedback = None
    updated_rework_count = state.rework_count or 0
    
    # Try to extract QA result from messages
    if qa_result.get("messages"):
        for msg in reversed(qa_result["messages"]):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            qa_passed = parsed.get("qa_result")
                            qa_feedback = parsed.get("qa_feedback")
                            if "rework_count" in parsed:
                                updated_rework_count = parsed.get("rework_count", updated_rework_count)
                            break
                    except (json.JSONDecodeError, TypeError):
                        # Check if content mentions pass/fail
                        content_lower = content.lower()
                        if "qa_result" in content_lower or "passed" in content_lower or "failed" in content_lower:
                            # Try to extract boolean from text
                            if "true" in content_lower or "passed" in content_lower:
                                qa_passed = True
                            elif "false" in content_lower or "failed" in content_lower:
                                qa_passed = False
                                qa_feedback = content
                            break
    
    # If QA passed or max reworks reached, go to final_results node
    # Otherwise, route back to the task type agent for rework
    if qa_passed is True:
        print(f"[QA] QA passed. Proceeding to final_results.")
        return {
            "qa_result": True,
            "qa_feedback": qa_feedback or "QA passed successfully",
            "next": "final_results",
            "next_node": "qa_agent",
            "campaign_diagnoses": diagnoses,
            "rework_count": updated_rework_count
        }
    elif updated_rework_count >= 3:
        print(f"[QA] Max reworks ({updated_rework_count}) reached. Using latest diagnoses and proceeding to final_results.")
        return {
            "qa_result": False,
            "qa_feedback": qa_feedback or "Max reworks reached, using latest diagnoses",
            "next": "final_results",
            "next_node": "qa_agent",
            "campaign_diagnoses": diagnoses,
            "rework_count": updated_rework_count
        }
    else:
        # Need rework - route back to the task type agent
        updated_rework_count += 1
        
        # Determine which agent to route back to based on campaign_brief.task_type
        task_type_agent = "new_creative_agent"  # default
        if state.campaign_brief and state.campaign_brief.task_type:
            task_type_lower = state.campaign_brief.task_type.lower().strip()
            if task_type_lower == "theme":
                task_type_agent = "theme_agent"
            elif task_type_lower == "new creative":
                task_type_agent = "new_creative_agent"
            elif task_type_lower in ["rework", "campaign update"]:
                task_type_agent = "campaign_update_agent"
        
        print(f"[QA] QA failed (attempt {updated_rework_count}/3). Routing back to {task_type_agent} for rework.")
        print(f"[QA] Feedback: {qa_feedback}")
        
        # Add feedback message for the agent to see
        updated_messages = state.messages.copy() if state.messages else []
        updated_messages.append({
            "role": "system",
            "content": f"QA Review (Attempt {updated_rework_count}/3): {qa_feedback or 'QA check failed. Please review and improve the diagnoses.'}\n\nIMPORTANT: The campaign brief is already in the workflow state. DO NOT call load_and_parse_spreadsheet - use the existing campaign_brief data from the previous messages."
        })
        
        return {
            "qa_result": False,
            "qa_feedback": qa_feedback,
            "next": task_type_agent,
            "next_node": "qa_agent",
            "campaign_diagnoses": diagnoses,
            "rework_count": updated_rework_count,
            "messages": updated_messages
        }


def final_results_node(state: AgentState) -> Dict[str, Any]:
    """
    Final results node that formats and delivers the workflow results in a friendly, structured format.
    
    Args:
        state: Current agent state with campaign brief and diagnoses
        
    Returns:
        Updated state with final_results containing structured output
    """
    import json
    
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
                "recommendations": diag.recommendations
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
        "qa_result": state.qa_result,
        "rework_count": state.rework_count or 0
    }
    
    # Create friendly message for the user
    friendly_message = f"""Great! I've completed the analysis of your Campaign Brief.

📊 Summary:
• It's a {task_type} Campaign Brief
• It has {total_campaigns} Campaign{'s' if total_campaigns != 1 else ''}
• Status Breakdown: {status_counts['critical']} critical, {status_counts['observed']} observed, {status_counts['passed']} passed

📋 Detailed Results (JSON format for frontend):
{json.dumps(final_results, indent=2)}
"""
    
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

    print(f"[Brief Creator] Result: {result}")
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
    workflow.add_node("qa_agent", qa_node)
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
    
    # All task type agents go to QA node after processing
    workflow.add_edge("theme_agent", "qa_agent")
    workflow.add_edge("new_creative_agent", "qa_agent")
    workflow.add_edge("campaign_update_agent", "qa_agent")
    
    # QA node routes based on result
    def route_from_qa(state: AgentState) -> str:
        """
        Route from QA node based on QA result and rework count.
        
        Returns:
            "final_results" if QA passed or max reworks reached
            The task type agent name if rework is needed
        """
        if state.next == "final_results":
            return "final_results"
        # Otherwise route back to the task type agent
        return state.next or "new_creative_agent"
    
    workflow.add_conditional_edges(
        "qa_agent",
        route_from_qa,
        {
            "final_results": "final_results",
            "theme_agent": "theme_agent",
            "new_creative_agent": "new_creative_agent",
            "campaign_update_agent": "campaign_update_agent"
        }
    )
    
    # Final results node always ends the workflow
    workflow.add_edge("final_results", END)
    
    return workflow.compile()
