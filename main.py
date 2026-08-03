"""
Main execution script for the Campaign Brief workflow.
"""
import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv
from graph.models import AgentState
from graph.langsmith_workflow import run_campaign_brief_workflow_traced
from graph.llm_provider import get_primary_provider
from graph.console_log import log_progress, strip_analytics_from_payload


def main():
    """
    Main execution function for the Campaign Brief workflow.
    """
    # Load environment variables
    load_dotenv()
    
    # Validate required provider credentials
    llm_provider = get_primary_provider()
    if llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required when LLM_PROVIDER=openai")
    elif llm_provider != "openai":
        raise ValueError(f"Unsupported LLM_PROVIDER: {llm_provider}")
    
    print("=" * 80)
    print("Campaign Brief Workflow - Starting Execution")
    print("=" * 80)
    
    print("\n[Setup] Workflow will be created at run time (see LangSmith trace: campaign_brief_workflow).")
    
    # Get spreadsheet path
    spreadsheet_path = "Campaigns/Steve Schmitt Group/steveschmittinchighland/2026-07-steveschmittinchighland-D-101822.xlsx"  # User will add the path here
    
    if not spreadsheet_path:
        print("\n⚠️  WARNING: spreadsheet_path is not set!")
        print("Please update the 'spreadsheet_path' variable in main.py")
        return
    
    # Validate spreadsheet file exists
    if not Path(spreadsheet_path).exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {spreadsheet_path}")
    
    print(f"\n[Input] Spreadsheet path: {spreadsheet_path}")
    
    # Initialize the workflow state
    initial_state = AgentState(
        messages=[
            {
                "role": "user",
                "content": f"Please process the campaign brief spreadsheet at: {spreadsheet_path}"
            }
        ],
        next_node="brief_creator"
    )
    
    print("\n[Execution] Starting workflow execution...")
    print("-" * 80)
    
    try:
        # Run the workflow
        # Note: The workflow nodes will print their own progress messages
        print("\n[Workflow] Executing workflow...")
        print("(Progress messages from workflow nodes will appear below)")
        print("-" * 80)
        
        # Invoke the workflow (runs to completion); root trace + compact I/O in LangSmith
        final_state = run_campaign_brief_workflow_traced(
            initial_state.model_dump(),
            spreadsheet_path,
        )
        
        print("\n" + "=" * 80)
        print("[Execution] Workflow completed successfully!")
        print("=" * 80)
        
        # Display final results
        # Handle both dict and Pydantic model responses
        final_results = final_state.get("final_results")
        if final_results:
            log_progress("\nFINAL RESULTS:")
            log_progress("-" * 80)

            # Convert to dict if it's a Pydantic model
            if hasattr(final_results, "model_dump"):
                final_results = final_results.model_dump()
            elif hasattr(final_results, "dict"):
                final_results = final_results.dict()

            # Console: user-facing payload only (metrics go to workflow_metrics.jsonl)
            display_results = strip_analytics_from_payload(final_results)
            print(json.dumps(display_results, indent=2, ensure_ascii=False))
            metrics_file = os.getenv("WORKFLOW_METRICS_FILE", "workflow_metrics.jsonl")
            log_progress(f"\nRun metrics logged to: {metrics_file}")
            
        else:
            print("\n⚠️  No final_results found in the workflow output.")
            print("Final state keys:", list(final_state.keys()))
            
            # Print campaign diagnoses if available
            diagnoses = final_state.get("campaign_diagnoses")
            if diagnoses:
                print("\n📋 Campaign Diagnoses:")
                diagnoses_list = []
                for diag in diagnoses:
                    if hasattr(diag, "model_dump"):
                        diagnoses_list.append(diag.model_dump())
                    elif hasattr(diag, "dict"):
                        diagnoses_list.append(diag.dict())
                    else:
                        diagnoses_list.append(diag)
                print(json.dumps(diagnoses_list, indent=2, ensure_ascii=False))
        
        # Print summary statistics
        brief = final_state.get("campaign_brief")
        if brief:
            # Handle Pydantic model
            if hasattr(brief, "task_type"):
                task_type = brief.task_type
                total_campaigns = len(brief.campaigns) if hasattr(brief, "campaigns") else 0
            elif isinstance(brief, dict):
                task_type = brief.get("task_type", "N/A")
                total_campaigns = len(brief.get("campaigns", []))
            else:
                task_type = "N/A"
                total_campaigns = 0
            
            print(f"\n📊 Summary:")
            print(f"  • Task Type: {task_type}")
            print(f"  • Total Campaigns: {total_campaigns}")
            print(f"  • QA Result: {final_state.get('qa_result', 'N/A')}")
            print(f"  • Rework Count: {final_state.get('rework_count', 0)}")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ ERROR: Workflow execution failed")
        print("=" * 80)
        print(f"\nError details: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

