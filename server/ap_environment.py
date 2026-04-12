"""server/ap_environment.py — APEnvironment core logic. No external deps beyond stdlib+pydantic."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional

# Dual-import: works both as `server.ap_environment` (Docker) and relative (in-repo)
try:
    from models import APAction, APObservation            # Docker: PYTHONPATH=/app
except ImportError:
    from ..models import APAction, APObservation          # type: ignore  # in-repo

try:
    from data_generator import build_easy, build_medium, build_hard  # Docker
except ImportError:
    from ..data_generator import build_easy, build_medium, build_hard  # type: ignore

try:
    from grader import grade_task                         # Docker
except ImportError:
    from ..grader import grade_task                       # type: ignore

HIGH_VALUE = 7500.0
DONE_STATUSES = {
    "validated", "flagged_fraud", "flagged_duplicate",
    "approved", "awaiting_approval", "rejected", "posted",
}
VALID_TOOLS = {
    "validate_data", "flag_fraud", "mark_duplicate",
    "route_approval", "post_ledger", "reject", "request_info",
}
MAX_STEPS = {
    "single_invoice_validation": 20,
    "fraud_duplicate_detection": 35,
    "full_ap_cycle": 50,
}
TASK_PROMPTS = {
    "single_invoice_validation": (
        "You are an AP clerk. ONE invoice needs processing. "
        "Step 1: call validate_data with invoice_id, amount, vendor_id. "
        "Step 2: call post_ledger with invoice_id, gl_code (from the PO), amount. "
        "Only reject if the PO is missing or the vendor does not match."
    ),
    "fraud_duplicate_detection": (
        "You are an AP fraud analyst. Process 3 invoices. "
        "If vendor_name starts with FAKE or amount is a large round number with no matching PO → flag_fraud. "
        "If the invoice looks identical to another you already processed → mark_duplicate. "
        "For legitimate invoices → validate_data then post_ledger."
    ),
    "full_ap_cycle": (
        "You are an AP manager. Process 5 invoices. "
        "For each: validate_data first. "
        "If amount > 7500 and valid → route_approval with approver_id from the matching PO. "
        "If FAKE vendor or suspicious round amount with no PO → flag_fraud. "
        "If duplicate of a previous invoice → mark_duplicate. "
        "After approval (or for low-value) → post_ledger with the gl_code from the PO."
    ),
}


class APEnvironment:
    """Accounts Payable Workflow Environment — step() / reset() / state()."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._task_id   = "single_invoice_validation"
        self._step      = 0
        self._max_steps = 20
        self._done      = False
        self._invoices:  List[Dict] = []
        self._pos:       List[Dict] = []
        self._ledger:    List[Dict] = []
        self._approvals: List[Dict] = []
        self._last_result: Optional[str] = None
        self._action_counts: Dict[str, int] = {}
        self._cumulative_reward: float = 0.0

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self, task_id: str = "single_invoice_validation",
              seed: Optional[int] = None) -> APObservation:
        if task_id not in TASK_PROMPTS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {list(TASK_PROMPTS.keys())}"
            )
        s = seed if seed is not None else self.seed
        self._task_id   = task_id
        self._step      = 0
        self._max_steps = MAX_STEPS[task_id]
        self._done      = False
        self._ledger    = []
        self._approvals = []
        self._action_counts = {}
        self._last_result   = None
        self._cumulative_reward = 0.0

        builders = {
            "single_invoice_validation": build_easy,
            "fraud_duplicate_detection": build_medium,
            "full_ap_cycle":             build_hard,
        }
        scenario = builders[task_id](s)
        self._invoices = scenario["invoices"]
        self._pos      = scenario["purchase_orders"]
        return self._obs()

    def step(self, action: APAction):
        if self._done:
            raise RuntimeError("Episode is done — call reset() first.")
        self._step += 1
        tool = (action.tool_call or "").strip().lower()
        self._action_counts[tool] = self._action_counts.get(tool, 0) + 1

        reward, detail, breakdown = self._dispatch(tool, action.parameters or {})
        self._cumulative_reward += reward

        all_done = all(i.get("status") in DONE_STATUSES for i in self._invoices)
        self._done = all_done or (self._step >= self._max_steps)

        obs = self._obs()
        obs.done = self._done
        return obs, reward, self._done, {
            "step":              self._step,
            "detail":            detail,
            "breakdown":         breakdown,
            "task_id":           self._task_id,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "remaining":         sum(1 for i in self._invoices
                                     if i.get("status") == "pending"),
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task_id":           self._task_id,
            "step_count":        self._step,
            "max_steps":         self._max_steps,
            "done":              self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "invoices":          self._invoices,
            "purchase_orders":   self._pos,
            "ledger_entries":    self._ledger,
            "approvals_history": self._approvals,
            "action_counts":     self._action_counts,
            "last_action_result": self._last_result,
        }

    # ── Observation builder ────────────────────────────────────────────────

    def _obs(self) -> APObservation:
        pending   = [i for i in self._invoices if i.get("status") == "pending"]
        processed = len(self._invoices) - len(pending)
        return APObservation(
            task_id            = self._task_id,
            task_prompt        = TASK_PROMPTS[self._task_id],
            step_number        = self._step,
            current_invoice    = pending[0] if pending else None,
            pending_invoices   = pending,
            purchase_orders    = self._pos,
            ledger_entries     = self._ledger,
            approvals_history  = self._approvals,
            last_action_result = self._last_result,
            available_actions  = sorted(VALID_TOOLS),
            invoices_processed = processed,
            total_invoices     = len(self._invoices),
            done               = self._done,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _inv(self, iid: str) -> Optional[Dict]:
        return next((i for i in self._invoices if i.get("invoice_id") == iid), None)

    def _po(self, po_num: str) -> Optional[Dict]:
        return next((p for p in self._pos if p.get("po_number") == po_num), None)

    def _loop_penalty(self) -> float:
        return -0.05 if sum(self._action_counts.values()) > 20 else 0.0

    def _dispatch(self, tool: str, params: dict):
        if tool not in VALID_TOOLS:
            self._last_result = f"ERROR: Unknown tool '{tool}'."
            return -0.1, "unknown_tool", {}
        handlers = {
            "validate_data":   self._do_validate,
            "flag_fraud":      self._do_flag_fraud,
            "mark_duplicate":  self._do_mark_dup,
            "route_approval":  self._do_route,
            "post_ledger":     self._do_post,
            "reject":          self._do_reject,
            "request_info":    self._do_info,
        }
        r, d, b = handlers[tool](params)
        lp = self._loop_penalty()
        return max(-1.0, min(1.0, r + lp)), d, b

    # ── Action handlers ────────────────────────────────────────────────────

    def _do_validate(self, p: dict):
        inv = self._inv(p.get("invoice_id", ""))
        if not inv:
            self._last_result = f"ERROR: invoice_id '{p.get('invoice_id')}' not found."
            return -0.2, "not_found", {"hallucination": -0.2}
        if inv.get("status") != "pending":
            self._last_result = f"INFO: {inv['invoice_id']} already {inv['status']}."
            return 0.0, "already_processed", {}

        po = self._po(inv.get("po_number", ""))
        score, issues = 0.0, []
        if po:
            score += 0.1
            if inv.get("vendor_id") == po.get("vendor_id"):
                score += 0.1
            else:
                issues.append("vendor_id mismatch")
            ratio = abs(inv.get("amount", 0) - po.get("approved_amount", 0))
            if ratio / max(po.get("approved_amount", 1), 1) <= 0.05:
                score += 0.1
            else:
                issues.append("amount deviation >5%")
        else:
            issues.append(f"PO {inv.get('po_number')} not found")

        inv["status"] = "validated"
        self._last_result = (
            f"VALIDATED {inv['invoice_id']}: "
            + ("; ".join(issues) if issues else "All fields OK.")
        )
        return score, "validated", {"field_bonus": score}

    def _do_flag_fraud(self, p: dict):
        inv = self._inv(p.get("invoice_id", ""))
        if not inv:
            self._last_result = f"ERROR: invoice_id '{p.get('invoice_id')}' not found."
            return -0.2, "not_found", {}
        if inv.get("is_fraudulent"):
            inv["status"] = "flagged_fraud"
            self._last_result = f"CORRECT FRAUD FLAG: {inv['invoice_id']}. Signals: {inv.get('fraud_signals')}"
            return 0.3, "fraud_caught", {"fraud_bonus": 0.3}
        self._last_result = f"FALSE POSITIVE: {inv['invoice_id']} is not fraudulent."
        return -0.1, "false_positive", {}

    def _do_mark_dup(self, p: dict):
        inv = self._inv(p.get("invoice_id", ""))
        if not inv:
            self._last_result = f"ERROR: invoice_id '{p.get('invoice_id')}' not found."
            return -0.2, "not_found", {}
        if inv.get("is_duplicate"):
            inv["status"] = "flagged_duplicate"
            self._last_result = f"CORRECT DUPLICATE: {inv['invoice_id']} blocked."
            return 0.3, "dup_caught", {"dup_bonus": 0.3}
        self._last_result = f"FALSE POSITIVE: {inv['invoice_id']} is not a duplicate."
        return -0.1, "false_positive", {}

    def _do_route(self, p: dict):
        inv = self._inv(p.get("invoice_id", ""))
        if not inv:
            self._last_result = f"ERROR: invoice_id '{p.get('invoice_id')}' not found."
            return -0.2, "not_found", {}
        if inv.get("status") not in ("pending", "validated"):
            self._last_result = f"INFO: {inv['invoice_id']} already {inv['status']}."
            return 0.0, "already_processed", {}

        po       = self._po(inv.get("po_number", ""))
        expected = po.get("approver_id", "") if po else ""
        given    = p.get("approver_id", "")
        correct  = given == expected

        if inv.get("amount", 0) >= HIGH_VALUE:
            inv["status"] = "approved" if correct else "awaiting_approval"
            self._approvals.append({
                "invoice_id":  inv["invoice_id"],
                "approver_id": given,
                "decision":    "approved" if correct else "pending",
                "timestamp":   datetime.utcnow().isoformat(),
            })
            bonus = 0.3 if correct else 0.1
            self._last_result = (
                f"{'CORRECT' if correct else 'PARTIAL'} ROUTING: "
                f"{inv['invoice_id']} → {given}. "
                f"{'Approved.' if correct else f'Expected {expected}.'}"
            )
        else:
            bonus = 0.0
            self._last_result = (
                f"INFO: {inv['invoice_id']} amount {inv.get('amount')} "
                f"is below threshold — no approval needed."
            )
        return bonus, "routed", {"approval_bonus": bonus}

    def _do_post(self, p: dict):
        inv = self._inv(p.get("invoice_id", ""))
        if not inv:
            self._last_result = f"ERROR: invoice_id '{p.get('invoice_id')}' not found."
            return -0.2, "not_found", {}

        status = inv.get("status", "")
        if status in ("flagged_fraud", "flagged_duplicate", "rejected"):
            self._last_result = f"BLOCKED: Cannot post {inv['invoice_id']} — status is {status}."
            return -0.1, "blocked", {}
        if status == "pending":
            self._last_result = f"ERROR: Validate {inv['invoice_id']} before posting."
            return -0.05, "needs_validation", {}
        if inv.get("amount", 0) >= HIGH_VALUE and status != "approved":
            self._last_result = (
                f"BLOCKED: {inv['invoice_id']} needs approval first "
                f"(amount: {inv.get('amount')})."
            )
            return -0.1, "needs_approval", {}
        if status == "posted":
            self._last_result = f"WARN: {inv['invoice_id']} already posted."
            return 0.0, "already_posted", {}

        po     = self._po(inv.get("po_number", ""))
        exp_gl = po.get("gl_code", "GL-9999") if po else "GL-9999"
        gl     = p.get("gl_code", exp_gl)
        ok_gl  = gl == exp_gl

        self._ledger.append({
            "entry_id":    f"LE-{len(self._ledger)+1:04d}",
            "invoice_id":  inv["invoice_id"],
            "gl_code":     gl,
            "amount":      inv["amount"],
            "date":        datetime.utcnow().date().isoformat(),
            "description": f"{inv['vendor_name']} / {inv['invoice_id']}",
        })
        inv["status"] = "posted"
        bonus = 0.5 if ok_gl else 0.2
        self._last_result = (
            f"POSTED: {inv['invoice_id']} (GL:{gl}). "
            f"{'Correct GL!' if ok_gl else f'Wrong GL — expected {exp_gl}.'}"
        )
        return bonus, "posted", {"post_bonus": bonus}

    def _do_reject(self, p: dict):
        inv = self._inv(p.get("invoice_id", ""))
        if not inv:
            self._last_result = f"ERROR: invoice_id '{p.get('invoice_id')}' not found."
            return -0.2, "not_found", {}
        inv["status"] = "rejected"
        if inv.get("is_fraudulent") or inv.get("is_duplicate"):
            self._last_result = f"CORRECT REJECT: {inv['invoice_id']}."
            return 0.2, "correct_reject", {}
        self._last_result = f"BAD REJECT: {inv['invoice_id']} is a valid invoice!"
        return -0.3, "wrong_reject", {}

    def _do_info(self, p: dict):
        self._last_result = (
            f"INFO REQUEST: field='{p.get('field')}' "
            f"for {p.get('invoice_id')}. No additional data in simulation."
        )
        return 0.0, "info_request", {}
