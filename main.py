"""
Main execution script for the Campaign Brief workflow.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from graph.workflow import create_campaign_workflow
from graph.models import AgentState


def main():
    """
    Main execution function for the Campaign Brief workflow.
    """
    # Load environment variables
    load_dotenv()
    
    # Validate required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    print("=" * 80)
    print("Campaign Brief Workflow - Starting Execution")
    print("=" * 80)
    
    # Create the workflow
    print("\n[Setup] Creating workflow...")
    workflow = create_campaign_workflow()
    print("[Setup] Workflow created successfully!")
    
    # Get spreadsheet path
    spreadsheet_path = "2025-12-novakgroup-A-25605547.xlsx"  # User will add the path here
    
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
        
        # Invoke the workflow (runs to completion)
        final_state = workflow.invoke(initial_state.model_dump())
        
        print("\n" + "=" * 80)
        print("[Execution] Workflow completed successfully!")
        print("=" * 80)
        
        # Display final results
        # Handle both dict and Pydantic model responses
        final_results = final_state.get("final_results")
        if final_results:
            print("\n📊 FINAL RESULTS:")
            print("-" * 80)
            
            # Convert to dict if it's a Pydantic model
            if hasattr(final_results, "model_dump"):
                final_results = final_results.model_dump()
            elif hasattr(final_results, "dict"):
                final_results = final_results.dict()
            
            # Pretty print the structured results
            print(json.dumps(final_results, indent=2, ensure_ascii=False))
            
            # Also save to file for easy access
            output_file = "workflow_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
            
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
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

