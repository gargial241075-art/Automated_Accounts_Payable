"""models.py — Typed Pydantic models. No openenv-core dependency."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class APAction(BaseModel):
    """Agent action: one tool call with parameters."""
    tool_call: str = Field(..., description=(
        "One of: validate_data | flag_fraud | mark_duplicate | "
        "route_approval | post_ledger | reject | request_info"
    ))
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None


class APObservation(BaseModel):
    """What the agent sees each step."""
    task_id: str = ""
    task_prompt: str = ""
    step_number: int = 0
    current_invoice: Optional[Dict[str, Any]] = None
    pending_invoices: List[Dict[str, Any]] = Field(default_factory=list)
    purchase_orders: List[Dict[str, Any]] = Field(default_factory=list)
    ledger_entries: List[Dict[str, Any]] = Field(default_factory=list)
    approvals_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_action_result: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)
    invoices_processed: int = 0
    total_invoices: int = 0
    done: bool = False
