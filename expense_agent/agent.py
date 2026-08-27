import base64
import json
from pydantic import BaseModel
from google.adk.workflow import Workflow, node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models import Gemini
import os
import google.auth

# Only use Vertex AI if the user has explicitly set up GCP credentials
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
except Exception:
    # Fallback to AI Studio API Key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# Configuration
class Config:
    THRESHOLD = 100.0
    RISK_MODEL = "gemini-3.1-flash-lite"

# Data Models
class Expense(BaseModel):
    amount: float
    submitter: str
    category: str
    description: str
    date: str

class RiskAssessment(BaseModel):
    risk_factors: str
    alert_raised: bool

from typing import Any
from google.genai import types

@node
def extract_expense(node_input: Any) -> Event:
    """Extracts expense details from the JSON event and routes based on amount."""
    data = None
    if isinstance(node_input, types.Content):
        if node_input.parts:
            data = node_input.parts[0].text
    elif isinstance(node_input, dict):
        data = node_input.get("data", node_input)
    else:
        data = node_input
        
    if not data:
        raise ValueError("Missing data in input")

    if isinstance(data, str):
        try:
            # Try parsing as plain JSON
            expense_dict = json.loads(data)
        except json.JSONDecodeError:
            # Try decoding base64 (Pub/Sub pattern)
            decoded = base64.b64decode(data).decode('utf-8')
            expense_dict = json.loads(decoded)
    elif isinstance(data, dict):
        expense_dict = data
    else:
        raise ValueError("Invalid data format for expense event.")

    expense = Expense(**expense_dict)
    
    # Save the expense into context state so human_review can access it later if needed
    state_delta = {"expense": expense.model_dump()}
    
    if expense.amount < Config.THRESHOLD:
        return Event(output=expense, route="auto_approve", state=state_delta)
    else:
        return Event(output=expense, route="review_required", state=state_delta)

import re

@node
def security_checkpoint(ctx: Context, node_input: Expense) -> Event:
    """Scrub PII and defend against prompt injection before LLM review."""
    original_desc = node_input.description
    redacted_categories = []
    
    # 1. Scrub SSNs (e.g., XXX-XX-XXXX)
    desc = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', original_desc)
    if desc != original_desc:
        redacted_categories.append("SSN")
        original_desc = desc
        
    # Scrub Credit Card numbers (13-19 digits, possibly with spaces/dashes)
    desc = re.sub(r'\b(?:\d[ -]*?){13,19}\b', '[REDACTED CC]', original_desc)
    if desc != original_desc:
        redacted_categories.append("Credit Card")
        
    node_input.description = desc
    # Remember redacted categories in state
    ctx.state["redacted_categories"] = redacted_categories
    
    # 2. Defend against prompt injection (heuristic check)
    suspicious_phrases = ["ignore previous instructions", "auto-approve", "bypass", "system prompt", "disregard"]
    is_injected = any(phrase in desc.lower() for phrase in suspicious_phrases)
    
    if is_injected:
        # Route straight to human review as a security event, bypassing LLM
        security_alert = RiskAssessment(
            risk_factors="SECURITY EVENT: Potential prompt injection detected in description.",
            alert_raised=True
        )
        return Event(output=security_alert, route="injection_detected")
    
    # Clean expense, continue to LLM reviewer
    return Event(output=node_input, route="clean")

@node
def auto_approve(node_input: Expense) -> str:
    """Instantly approves the expense."""
    return f"Auto-approved expense from {node_input.submitter} for ${node_input.amount}."

risk_reviewer = LlmAgent(
    name="risk_reviewer",
    model=Gemini(model=Config.RISK_MODEL),
    instruction="""You are a financial risk analyst.
Review the provided expense report for any risk factors (e.g., suspicious categories, unusually high amounts for the description).
Determine if an alert should be raised.""",
    output_schema=RiskAssessment
)

@node
async def human_review(ctx: Context, node_input: RiskAssessment):
    """Pauses for human-in-the-loop approval, displaying the risk assessment."""
    if not ctx.resume_inputs:
        expense_data = ctx.state.get("expense", {})
        redactions = ctx.state.get("redacted_categories", [])
        redaction_note = f"\n[Note: Redacted {', '.join(redactions)}]" if redactions else ""
        
        message = (
            f"Review required for expense from {expense_data.get('submitter')} "
            f"for ${expense_data.get('amount')}.{redaction_note}\n"
            f"Risk Assessment: {node_input.risk_factors}\n"
            f"Alert Raised: {node_input.alert_raised}\n"
            f"Please approve or reject."
        )
        yield RequestInput(interrupt_id="approval_decision", message=message)
        return
    
    decision = ctx.resume_inputs.get("approval_decision")
    yield Event(output=decision)

@node
def record_outcome(ctx: Context, node_input: str) -> str:
    """Records the final decision outcome."""
    expense_data = ctx.state.get("expense", {})
    return f"Outcome for {expense_data.get('submitter')}'s ${expense_data.get('amount')} expense: {node_input}"

# Wire the graph edges
workflow = Workflow(
    name="ambient_expense_agent",
    edges=[
        ('START', extract_expense),
        (extract_expense, {
            "auto_approve": auto_approve,
            "review_required": security_checkpoint
        }),
        (security_checkpoint, {
            "clean": risk_reviewer,
            "injection_detected": human_review
        }),
        (risk_reviewer, human_review),
        (human_review, record_outcome)
    ]
)

app = App(
    name="expense_agent",
    root_agent=workflow
)
