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
    headline: str = Field(default="", description="The headline of the campaign")
    offer: str = Field(default="", description="The offer of the campaign")
    body: str = Field(default="", description="The body of the offer")
    cta: str = Field(default="", description="The call to action of the offer")
    disclaimer: str = Field(default="", description="The disclaimer of the offer")

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
    grounding_evidence: List[str] = Field(
        default_factory=list,
        description="Short evidence snippets or source references used to ground this diagnosis."
    )


class FamilySimilarityCampaignMatch(BaseModel):
    target_campaign_id: str = Field(..., description="Campaign ID from the target brief")
    candidate_campaign_id: str = Field(..., description="Campaign ID from the candidate brief")
    similarity_score: float = Field(..., description="Weighted similarity score in [0, 1]")
    scoring_path: str = Field(
        default="content",
        description="Which scoring path was used: 'reference' (copy/refer ID) or 'content'",
    )
    pair_status: str = Field(
        default="none",
        description="Pairing decision: absolute | likely | review | none",
    )
    pair_basis: str = Field(
        default="content",
        description="Primary basis for the pair: reference | style_assets | content",
    )
    style_direction_similarity: float = Field(
        ...,
        description=(
            "Combined StyleDirection section score in [0, 1]: "
            "style direction / additional style / vehicle photography / assets"
        ),
    )
    style_fields_similarity: float = Field(
        default=0.0,
        description="Similarity across style text columns in [0, 1]",
    )
    asset_structure_similarity: float = Field(
        default=0.0,
        description="Asset column similarity in [0, 1]",
    )
    campaign_wording_similarity: float = Field(
        ...,
        description="Campaign structure + offer wording similarity in [0, 1]",
    )
    dealership_relationship: float = Field(
        ...,
        description="Dealership/OEM/group proximity score in [0, 1] (10% weight)",
    )
    reference_strength: float = Field(
        default=0.0,
        description="Copy/refer ID signal strength in [0, 1] when scoring_path=reference",
    )
    reference_id_boost: float = Field(
        default=0.0,
        description="Compatibility alias for reference_strength",
    )
    reference_boost_reasons: List[str] = Field(
        default_factory=list,
        description="Why a copy/refer ID signal was applied",
    )
    has_copy_refer_signal: bool = Field(
        default=False,
        description="True when copy/refer cue wording + matching A-/D- IDs were found",
    )
    match_reason: str = Field(
        default="",
        description="Direct narrative reason why these campaigns are considered paired"
    )
    evidence_points: List[str] = Field(
        default_factory=list,
        description="Short evidence bullets grounded in assets/style/offer fields"
    )


class FamilySimilarityCandidate(BaseModel):
    file_name: str = Field(..., description="Candidate spreadsheet filename")
    file_path: str = Field(..., description="Candidate spreadsheet local path")
    dealership_name: Optional[str] = Field(default=None, description="Dealership name parsed from candidate brief")
    task_type: Optional[str] = Field(default=None, description="Task type parsed from candidate brief")
    file_similarity_score: float = Field(..., description="Aggregate file-level similarity score in [0, 1]")
    strongest_matches: List[FamilySimilarityCampaignMatch] = Field(
        default_factory=list,
        description="Top campaign matches for this candidate file"
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
    diagnoses_json_path: Optional[str] = Field(
        default=None,
        description="Path to the diagnoses JSON file created by the diagnosis formatter"
    )
    family_similarity: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload for the family similarity branch (toggle route 0)"
    )