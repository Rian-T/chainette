from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
import datetime

from chainette import (
    Step,
    Chain,
    Branch,
    SamplingParams,
    register_engine,
    ApplyNode,
)

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

class SupportTicketInput(BaseModel):
    """Raw incoming support ticket"""
    ticket_id: str = Field(..., description="Unique identifier for the support ticket")
    customer_email: str = Field(..., description="Email address of the customer")
    subject: str = Field(..., description="Subject line of the ticket")
    description: str = Field(..., description="Full text description of the customer's issue")
    timestamp: str = Field(..., description="ISO format timestamp when ticket was created")

class TicketClassification(BaseModel):
    """Initial ticket classification and metadata extraction"""
    ticket_id: str
    issue_type: Literal["technical", "billing", "account", "feature_request", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    estimated_resolution_time: str = Field(..., description="Estimated time to resolve (e.g., '2 hours', '1 day')")
    key_points: List[str] = Field(..., description="List of key points extracted from the ticket")

class TechnicalAnalysis(BaseModel):
    """Technical analysis for technical support tickets"""
    ticket_id: str
    system_components: List[str] = Field(..., description="System components involved in the issue")
    possible_causes: List[str] = Field(..., description="Potential causes of the technical problem")
    suggested_solutions: List[str] = Field(..., description="Recommended technical solutions")
    references: Optional[List[str]] = Field(None, description="Links to relevant documentation")

class BillingAnalysis(BaseModel):
    """Billing analysis for billing/payment related tickets"""
    ticket_id: str
    transaction_details: Optional[dict] = Field(None, description="Relevant transaction details if identified")
    billing_issues: List[str] = Field(..., description="Specific billing issues identified")
    recommended_actions: List[str] = Field(..., description="Recommended actions for billing issues")
    refund_recommended: bool = Field(..., description="Whether a refund is recommended")
    refund_amount: Optional[float] = Field(None, description="Suggested refund amount if applicable")

class DraftResponse(BaseModel):
    """Initial draft response to the customer"""
    ticket_id: str
    response_text: str = Field(..., description="Draft response text to send to customer")
    follow_up_needed: bool = Field(..., description="Whether a follow-up is required")
    internal_notes: Optional[str] = Field(None, description="Notes for internal reference")

class SentimentAnalysis(BaseModel):
    """Customer sentiment analysis"""
    ticket_id: str
    sentiment: Literal["very_negative", "negative", "neutral", "positive", "very_positive"]
    urgency_level: int = Field(..., ge=1, le=10, description="Detected urgency level (1-10)")
    key_concerns: List[str] = Field(..., description="Key customer concerns detected")
    satisfaction_risk: Literal["low", "medium", "high"] = Field(..., description="Risk of customer dissatisfaction")

class FinalTicketProcessing(BaseModel):
    """Complete ticket processing result"""
    ticket_id: str
    original_subject: str
    classification: TicketClassification
    technical_analysis: Optional[TechnicalAnalysis] = None
    billing_analysis: Optional[BillingAnalysis] = None
    draft_response: DraftResponse
    sentiment: SentimentAnalysis
    suggested_tags: List[str] = Field(..., description="Suggested tags for the ticket")

class FinalizeInput(BaseModel):
    """Combined input for the finalize ticket step"""
    original_subject: str = Field(..., description="Original subject of the ticket")
    classification: TicketClassification = Field(..., description="Ticket classification data")
    sentiment: SentimentAnalysis = Field(..., description="Customer sentiment analysis")
    technical_analysis: Optional[TechnicalAnalysis] = Field(None, description="Technical analysis if available")
    billing_analysis: Optional[BillingAnalysis] = Field(None, description="Billing analysis if available") 
    draft_response: DraftResponse = Field(..., description="Draft response to the customer")

# ---------------------------------------------------------------------------
# Engine Registration
# ---------------------------------------------------------------------------

register_engine(
    name="llama3",
    model="meta-llama/Llama-3.2-3B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.95,
    lazy=True,
    max_model_len=2048
)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

classify = Step(
    id="classify",
    name="Classify Ticket",
    input_model=SupportTicketInput,
    output_model=TicketClassification,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.1),
    system_prompt=(
        "You are an expert customer support ticket classifier. Analyze the ticket to determine "
        "the issue type, priority level, and extract key points that will help resolve the issue. "
        "Estimate how long resolution might take based on the complexity of the issue."
    ),
    user_prompt="""
Please classify this support ticket:

TICKET ID: {{ticket_id}}
SUBJECT: {{subject}}
DESCRIPTION: {{description}}
TIMESTAMP: {{timestamp}}

Determine the issue type, priority, estimated resolution time, and extract key points.
""",
)

technical_analysis = Step(
    id="tech_analysis",
    name="Technical Analysis",
    input_model=TicketClassification,
    output_model=TechnicalAnalysis,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.2),
    system_prompt=(
        "You are a technical support specialist. Analyze technical support tickets to identify "
        "affected system components, possible causes, and recommend technical solutions. "
        "Include references to documentation where appropriate."
    ),
    user_prompt="""
Analyze this technical support ticket:

TICKET ID: {{ticket_id}}
PRIORITY: {{priority}}
KEY POINTS:
{% for point in key_points %}
- {{point}}
{% endfor %}

Provide detailed technical analysis including affected system components, possible causes, 
and recommended solutions.
""",
)

billing_analysis = Step(
    id="billing_analysis",
    name="Billing Analysis",
    input_model=TicketClassification,
    output_model=BillingAnalysis,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.2),
    system_prompt=(
        "You are a billing and payments specialist. Analyze billing-related tickets to identify "
        "transaction details, specific issues, and recommend appropriate actions. Determine if "
        "a refund is warranted and suggest an amount if applicable."
    ),
    user_prompt="""
Analyze this billing support ticket:

TICKET ID: {{ticket_id}}
PRIORITY: {{priority}}
KEY POINTS:
{% for point in key_points %}
- {{point}}
{% endfor %}

Provide detailed billing analysis including transaction details, specific issues identified, 
and recommended actions. Determine if a refund is appropriate and suggest an amount if needed.
""",
)

draft_technical_response = Step(
    id="draft_tech_response",
    name="Draft Technical Response",
    input_model=TechnicalAnalysis,
    output_model=DraftResponse,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.4),
    system_prompt=(
        "You are a helpful, professional technical support representative. Draft a clear, concise "
        "response to the customer addressing their technical issue. Be empathetic and solution-focused. "
        "Provide clear next steps and indicate if follow-up will be needed."
    ),
    user_prompt="""
Draft a response for this technical issue:

TICKET ID: {{ticket_id}}
POSSIBLE CAUSES:
{% for cause in possible_causes %}
- {{cause}}
{% endfor %}
SUGGESTED SOLUTIONS:
{% for solution in suggested_solutions %}
- {{solution}}
{% endfor %}

Create a professional, helpful response that addresses the technical issue and offers clear next steps.
""",
)

draft_billing_response = Step(
    id="draft_billing_response",
    name="Draft Billing Response",
    input_model=BillingAnalysis,
    output_model=DraftResponse,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.4),
    system_prompt=(
        "You are a helpful, professional billing support representative. Draft a clear, concise "
        "response to the customer addressing their billing issue. Be empathetic and solution-focused. "
        "If a refund is being offered, clearly state this and provide any necessary follow-up steps."
    ),
    user_prompt="""
Draft a response for this billing issue:

TICKET ID: {{ticket_id}}
BILLING ISSUES:
{% for issue in billing_issues %}
- {{issue}}
{% endfor %}
RECOMMENDED ACTIONS:
{% for action in recommended_actions %}
- {{action}}
{% endfor %}
REFUND RECOMMENDED: {{refund_recommended}}
{% if refund_amount %}REFUND AMOUNT: ${{refund_amount}}{% endif %}

Create a professional, helpful response that addresses the billing issue and offers clear next steps.
""",
)

sentiment_analysis = Step(
    id="sentiment",
    name="Sentiment Analysis",
    input_model=SupportTicketInput,
    output_model=SentimentAnalysis,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.1),
    system_prompt=(
        "You are an expert in customer sentiment analysis. Analyze the support ticket to determine "
        "the customer's emotional state, urgency level, key concerns, and the risk of customer "
        "dissatisfaction if the issue is not resolved promptly and effectively."
    ),
    user_prompt="""
Analyze the sentiment in this support ticket:

SUBJECT: {{subject}}
DESCRIPTION: {{description}}

Determine the customer's sentiment, urgency level (1-10), key concerns, and satisfaction risk level.
""",
)

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def combine_inputs_for_finalize(data_dict: Dict[str, Any], ticket_input: SupportTicketInput) -> FinalizeInput:
    """Combines outputs from branches and other steps into a single FinalizeInput object."""
    # Extract the necessary components from the data dictionary
    tech_branch = data_dict.get("technical", {})
    bill_branch = data_dict.get("billing", {})
    
    # Get classification (assuming it's stored with key "classify" from earlier step)
    classification = data_dict.get("classify")
    
    # Get sentiment analysis (assuming it's stored with key "sentiment" from earlier step)
    sentiment = data_dict.get("sentiment")
    
    # Get draft response from the appropriate branch
    draft_response = None
    if "draft_tech_response" in tech_branch:
        draft_response = tech_branch.get("draft_tech_response")
    elif "draft_billing_response" in bill_branch:
        draft_response = bill_branch.get("draft_billing_response")
    
    # Get technical and billing analysis if available
    tech_analysis = tech_branch.get("tech_analysis")
    bill_analysis = bill_branch.get("billing_analysis")
    
    return FinalizeInput(
        original_subject=ticket_input.subject,
        classification=classification,
        sentiment=sentiment,
        technical_analysis=tech_analysis,
        billing_analysis=bill_analysis,
        draft_response=draft_response
    )

# Add adapter functions for branches
def extract_classification(data_dict: Dict[str, Any]) -> TicketClassification:
    """Extract classification data from dictionary output of parallel steps."""
    # The dictionary from parallel steps should have key "classify" containing the classification
    classification = data_dict.get("classify")
    if not classification:
        raise ValueError("Classification data not found in input dictionary")
    return classification

# Fix ApplyNode instantiations by removing unsupported parameters
finalize_data_prep = ApplyNode(
    id="prepare_finalize_data",
    name="Prepare Data for Finalize",
    fn=combine_inputs_for_finalize,
    kwargs={"ticket_input": "input"}
)

# Create adapter nodes for branches without the unsupported parameters
tech_adapter = ApplyNode(
    id="technical_adapter",
    name="Technical Input Adapter",
    fn=extract_classification,
    kwargs={}
)

billing_adapter = ApplyNode(
    id="billing_adapter",
    name="Billing Input Adapter",
    fn=extract_classification,
    kwargs={}
)

# ---------------------------------------------------------------------------
# Branches and Chain
# ---------------------------------------------------------------------------

technical_branch = Branch(
    name="technical",
    steps=[
        tech_adapter,      # Add adapter as first step
        technical_analysis,
        draft_technical_response,
    ],
    emoji="🔧"
)

billing_branch = Branch(
    name="billing",
    steps=[
        billing_adapter,   # Add adapter as first step
        billing_analysis,
        draft_billing_response,
    ],
    emoji="💰"
)

finalize_ticket = Step(
    id="finalize",
    name="Finalize Ticket",
    input_model=FinalizeInput,
    output_model=FinalTicketProcessing,
    engine_name="llama3",
    sampling=SamplingParams(temperature=0.1),
    system_prompt=(
        "You are a customer support coordinator responsible for finalizing ticket processing. "
        "Integrate all analyses and draft responses into a comprehensive ticket processing result. "
        "Add appropriate tags that will help with ticket routing, reporting, and knowledge management."
    ),
    user_prompt="""
Finalize this support ticket processing:

TICKET ID: {{classification.ticket_id}}
ORIGINAL SUBJECT: {{original_subject}}
CLASSIFICATION: Issue Type: {{classification.issue_type}}, Priority: {{classification.priority}}
SENTIMENT: {{sentiment.sentiment}}, Urgency: {{sentiment.urgency_level}}/10

{% if technical_analysis %}
TECHNICAL ANALYSIS AVAILABLE
{% endif %}

{% if billing_analysis %}
BILLING ANALYSIS AVAILABLE
{% endif %}

DRAFT RESPONSE AVAILABLE: Follow-up needed: {{draft_response.follow_up_needed}}

Create a final processing record and suggest appropriate tags for this ticket.
""",
)

complex_chain = Chain(
    name="Support Ticket Processing",
    emoji="🎫",
    steps=[
        [
            classify,
            sentiment_analysis,
        ],
        [
            technical_branch,
            billing_branch,
        ],
        finalize_data_prep,
        finalize_ticket,
    ],
    batch_size=2,
)

if __name__ == "__main__":
    sample_tickets = [
        SupportTicketInput(
            ticket_id="TICKET_001",
            customer_email="user1@example.com",
            subject="Cannot connect to VPN",
            description="Hi team, I\'ve been trying to connect to the company VPN for the last hour but it keeps failing with error code 789. My internet is working fine. I need this to access project files urgently!",
            timestamp="2025-05-06T10:00:00Z"
        ),
        SupportTicketInput(
            ticket_id="TICKET_002",
            customer_email="user2@example.com",
            subject="Incorrect charge on invoice INV-2025-04-123",
            description="Hello, I was reviewing my latest invoice (INV-2025-04-123) and I see a charge for \'Premium Widget Service\' for $99.99 that I did not subscribe to. Can you please investigate and refund this amount?",
            timestamp="2025-05-06T11:30:00Z"
        ),
        SupportTicketInput(
            ticket_id="TICKET_003",
            customer_email="user3@example.com",
            subject="Question about product X features",
            description="Good morning, I\'m interested in your Product X. Could you tell me if it supports integration with ZapPlatform and what the storage limits are for the basic plan?",
            timestamp="2025-05-06T12:15:00Z"
        ),
    ]

    results_dataset_dict = complex_chain.run(
        inputs=sample_tickets,
        output_dir="output/complex_support_run",
        fmt="jsonl"
    )
    
    # Print just the finalized ticket for the first result as an example
    if "finalize" in results_dataset_dict:
        first_result = results_dataset_dict["finalize"][0]
        print("\n===== EXAMPLE FINALIZED TICKET =====")
        print(f"Ticket ID: {first_result.ticket_id}")
        print(f"Subject: {first_result.original_subject}")
        print(f"Issue Type: {first_result.classification.issue_type}")
        print(f"Priority: {first_result.classification.priority}")
        print(f"Sentiment: {first_result.sentiment.sentiment}")
        print(f"Tags: {', '.join(first_result.suggested_tags)}")
        print(f"Draft Response: {first_result.draft_response.response_text[:100]}...")