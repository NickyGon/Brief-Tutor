"""
Pydantic models for the LangGraph agent workflow.
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class Assets(BaseModel):
    sl_bn_srp_da: str = Field(default="", description="The desktop assets for the campaign")
    sl_m_bn_m: str = Field(default="", description="The mobile assets for the campaign")
    facebook_assets: str = Field(default="", description="The Facebook assets for the campaign")
    instagram_assets: str = Field(default="", description="The Instagram assets for the campaign")
    google_assets: str = Field(default="", description="The Google assets for the campaign")
    ot_1: str = Field(default="", description="The first of six additional assets for the campaign")
    ot_2: str = Field(default="", description="The second of six additional assets for the campaign")
    ot_3: str = Field(default="", description="The third of six additional assets for the campaign")
    ot_4: str = Field(default="", description="The fourth of six additional assets for the campaign")
    ot_5: str = Field(default="", description="The fifth of six additional assets for the campaign")
    ot_6: str = Field(default="", description="The sixth of six additional assets for the campaign")

class OfferDetails(BaseModel):
    headline: str = Field(default="", description="The headline of the campaign", max_length=75)
    offer: str = Field(default="", description="The offer of the campaign", max_length=100)
    body: str = Field(default="", description="The body of the offer", max_length=10000)
    cta: str = Field(default="", description="The call to action of the offer", max_length=30)
    disclaimer: str = Field(default="", description="The disclaimer of the offer", max_length=10000)

class StyleDescriptions(BaseModel):
    asset_style_direction: str = Field(default="", description="A specific theme for the campaign's multiple assets to follow.")
    additional_style_information: str = Field(default="", description="Any extra info for the campaign's multiple assets to use.")
    vehicle_photography: str = Field(default="", description="The description of the vehicle photo type to be used")
    logos: str = Field(default="", description="The asset logos to be added to the campaign")

class Campaign(BaseModel):
    """A campaign object"""
    campaign_id: str = Field(..., description="The campaign's identifier, indicated by Content and the number")
    style_descriptions: StyleDescriptions = Field(..., description="The style descriptions")
    offer_details: OfferDetails = Field(..., description="The offer details")
    assets: Assets = Field(..., description="The assets for the campaign")

class CampaignBrief(BaseModel):
    """A Campaign brief spreadsheet"""
    spreadsheet_path: str = Field(
        description="The path to the campaign brief spreadsheet"
    )
    task_type: str = Field(
        description="The type of campaign grooming task to be performed"
    )
    asset_summary: Optional[str] = Field(default=None, description="A summary of the total number of assets per type found between all campaigns")
    dealership_name: Optional[str] = Field(default=None, description="The name of the car dealership the campaigns will be posted to")
    content_11_20: bool = Field(default=False, description="Whether the spreadsheet has more than one tab to be read")
    campaigns: List[Campaign] = Field(..., description="The campaigns to be processed")


class CampaignDiagnosis(BaseModel):
    """A single campaign diagnosis from a task type agent"""
    campaign_id: str = Field(
        ...,
        description="The identifier of the campaign"
    )
    status: Literal["critical", "observed", "passed"] = Field(
        ...,
        description="The status of the campaign evaluation - must be one of: critical, observed, or passed"
    )
    diagnosis: str = Field(
        ...,
        description="The evaluation/diagnosis for this campaign based on the RAG rules"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of issues found (if any)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommendations (if any)"
    )


class AgentState(BaseModel):
    """State that flows through the agent graph."""
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of messages in the conversation"
    )
    next: Optional[str] = Field(
        default=None,
        description="Next node to execute (set by router based on campaign_brief.task_type)"
    )
    next_node: Optional[str] = Field(
        default=None,
        description="Current node being executed (for logging purposes)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the agent"
    )
    campaign_brief: Optional[CampaignBrief] = Field(
        default=None,
        description="The parsed campaign brief from the spreadsheet (contains task_type for routing)"
    )
    campaign_diagnoses: Optional[List[CampaignDiagnosis]] = Field(
        default=None,
        description="List of campaign diagnoses from the task type agent"
    )
    rework_count: int = Field(
        default=0,
        description="Number of times rework has been requested (max 3)"
    )
    qa_result: Optional[bool] = Field(
        default=None,
        description="QA result: True if passed, False if failed"
    )
    qa_feedback: Optional[str] = Field(
        default=None,
        description="Feedback from QA agent if the diagnoses need rework"
    )
    final_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final structured results from the workflow"
    )
