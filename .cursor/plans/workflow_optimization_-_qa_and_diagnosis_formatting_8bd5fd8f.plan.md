---
name: Workflow Optimization - QA and Diagnosis Formatting
overview: Optimize the workflow by consolidating QA validation into task agents with a diagnosis formatter, and adding a document creator agent that generates a brief resume (focused on observed/critical campaigns) and a full diagnoses listing. The formatter will use RAG-based validation and format diagnoses into the Excel template, then pass diagnoses to the document creator.
todos:
  - id: update_task_agents
    content: Add self-validation instructions to theme_agent.yaml, new_creative_agent.yaml, and campaign_update_agent.yaml
    status: completed
  - id: create_diagnosis_formatter_node
    content: Create diagnosis_formatter_node in workflow.py that validates, formats to Excel, and keeps diagnoses in state
    status: completed
  - id: create_document_creator_agent
    content: Create document_creator_agent.yaml that generates brief resume and full diagnoses listing documents
    status: completed
  - id: create_document_creator_node
    content: Create document_creator_node in workflow.py that invokes the document creator agent
    status: completed
  - id: update_workflow_edges
    content: Update workflow edges to route Task Agents → Diagnosis Formatter → Document Creator → Final Results
    status: completed
  - id: add_document_storage
    content: Add functions to rag_ingestion.py for storing document creator outputs to Google Drive (if needed)
    status: completed
  - id: remove_qa_agent
    content: Remove qa_node from workflow.py and delete agents/qa_agent.yaml
    status: completed
---

