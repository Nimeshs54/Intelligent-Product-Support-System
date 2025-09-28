# feature_store/schema.py
from __future__ import annotations
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd

# ---------- Pydantic ticket schema (relaxed enums to accept real-world values) ----------
class Ticket(BaseModel):
    ticket_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    customer_id: Optional[str] = None
    customer_tier: Optional[str] = None
    organization_id: Optional[str] = None

    product: Optional[str] = None
    product_version: Optional[str] = None
    product_module: Optional[str] = None

    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    severity: Optional[str] = None

    channel: Optional[str] = None
    subject: str
    description: str
    error_logs: Optional[str] = None
    stack_trace: Optional[str] = None

    customer_sentiment: Optional[str] = None
    previous_tickets: Optional[int] = Field(default=None, ge=0)
    resolution: Optional[str] = None
    resolution_code: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_time_hours: Optional[float] = None
    resolution_attempts: Optional[int] = None

    agent_id: Optional[str] = None
    agent_experience_months: Optional[int] = None
    agent_specialization: Optional[str] = None
    agent_actions: Optional[List[str]] = None

    escalated: Optional[bool] = None
    escalation_reason: Optional[str] = None
    transferred_count: Optional[int] = None

    satisfaction_score: Optional[int] = Field(default=None, ge=1, le=5)
    feedback_text: Optional[str] = None
    resolution_helpful: Optional[bool] = None

    tags: Optional[List[str]] = None
    related_tickets: Optional[List[str]] = None
    kb_articles_viewed: Optional[List[str]] = None
    kb_articles_helpful: Optional[List[str]] = None

    environment: Optional[str] = None
    account_age_days: Optional[int] = None
    account_monthly_value: Optional[float] = None
    similar_issues_last_30_days: Optional[int] = None
    product_version_age_days: Optional[int] = None
    known_issue: Optional[bool] = None
    bug_report_filed: Optional[bool] = None
    resolution_template_used: Optional[str] = None
    auto_suggested_solutions: Optional[List[str]] = None
    auto_suggestion_accepted: Optional[bool] = None
    ticket_text_length: Optional[int] = None
    response_count: Optional[int] = None
    attachments_count: Optional[int] = None
    contains_error_code: Optional[bool] = None
    contains_stack_trace: Optional[bool] = None
    business_impact: Optional[str] = None
    affected_users: Optional[int] = None
    weekend_ticket: Optional[bool] = None
    after_hours: Optional[bool] = None
    language: Optional[str] = None
    region: Optional[str] = None


# ---------- Validation helpers ----------
REQUIRED_TEXT_FIELDS = ["subject", "description"]
REQUIRED_TIME_FIELDS = ["created_at"]

def validate_records_json(records: list[dict]) -> list[Ticket]:
    """Validate every record against the Pydantic model and return typed objects."""
    validated: list[Ticket] = []
    for r in records:
        validated.append(Ticket.model_validate(r))
    return validated


def dq_validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Lightweight data-quality checks (no external libs).
    Returns a JSON-serializable dict with overall success and per-check results.
    """
    checks = []

    # 1) Non-null required columns
    for col in (REQUIRED_TEXT_FIELDS + REQUIRED_TIME_FIELDS + ["ticket_id"]):
        if col in df.columns:
            nulls = int(df[col].isna().sum())
            passed = nulls == 0
            checks.append({
                "check": f"non_null:{col}",
                "passed": passed,
                "null_count": nulls
            })
        else:
            checks.append({"check": f"non_null:{col}", "passed": False, "error": "missing_column"})

    # 2) created_at <= updated_at (when both present)
    if {"created_at", "updated_at"}.issubset(df.columns):
        both = df.dropna(subset=["created_at", "updated_at"])
        bad = int((both["created_at"] > both["updated_at"]).sum())
        checks.append({
            "check": "time_order:created_at<=updated_at",
            "passed": bad == 0,
            "violations": bad
        })

    # 3) created_at <= resolved_at (when both present)
    if {"created_at", "resolved_at"}.issubset(df.columns):
        both = df.dropna(subset=["created_at", "resolved_at"])
        bad = int((both["created_at"] > both["resolved_at"]).sum())
        checks.append({
            "check": "time_order:created_at<=resolved_at",
            "passed": bad == 0,
            "violations": bad
        })

    # 4) satisfaction_score in [1,5] (when present)
    if "satisfaction_score" in df.columns:
        s = df["satisfaction_score"].dropna()
        bad = int(((s < 1) | (s > 5)).sum())
        checks.append({
            "check": "range:satisfaction_score_1_to_5",
            "passed": bad == 0,
            "violations": bad
        })

    success = all(c.get("passed", False) for c in checks)
    return {"success": success, "checks": checks, "row_count": int(len(df))}
