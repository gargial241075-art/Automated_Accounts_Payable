"""grader.py — Deterministic task graders, scores strictly in (0.0, 1.0)."""
from __future__ import annotations
from typing import Any, Dict


# Clamp to strictly (0, 1) — never 0.0 or 1.0 exactly
_MIN = 0.01
_MAX = 0.99


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return max(_MIN, min(_MAX, round(score, 4)))


def grade_task(state: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Entry point called by openenv.yaml grader path: grader.grade_task"""
    if task_id == "single_invoice_validation":
        return _easy(state)
    elif task_id == "fraud_duplicate_detection":
        return _medium(state)
    elif task_id == "full_ap_cycle":
        return _hard(state)
    return _r(_MIN, {}, f"Unknown task_id: {task_id}")


# ── Easy ───────────────────────────────────────────────────────────────────

def _easy(state: dict) -> dict:
    invoices = state.get("invoices", [])
    ledger   = state.get("ledger_entries", [])
    if not invoices:
        return _r(_MIN, {}, "No invoices in state.")

    inv    = invoices[0]
    status = inv.get("status", "")
    inv_id = inv.get("invoice_id")
    posted = any(e.get("invoice_id") == inv_id for e in ledger)

    correct_gl = False
    if posted:
        po_num = inv.get("po_number", "")
        po = next((p for p in state.get("purchase_orders", [])
                   if p.get("po_number") == po_num), None)
        exp_gl = po.get("gl_code", "") if po else ""
        entry  = next((e for e in ledger if e.get("invoice_id") == inv_id), None)
        correct_gl = bool(entry and entry.get("gl_code") == exp_gl)

    bd = {"status": status, "posted": posted, "correct_gl": correct_gl}
    if status == "posted" and correct_gl:
        return _r(0.98, bd, "Perfect: validated and posted to correct GL.")
    if status == "posted":
        return _r(0.70, bd, "Posted but wrong GL code.")
    if status in ("validated", "approved"):
        return _r(0.50, bd, "Validated but not posted.")
    if status == "rejected" and not inv.get("is_fraudulent") and not inv.get("is_duplicate"):
        return _r(0.02, bd, "Fail: valid invoice incorrectly rejected.")
    return _r(0.10, bd, f"Not fully processed (status={status}).")


# ── Medium ─────────────────────────────────────────────────────────────────

def _medium(state: dict) -> dict:
    invoices = state.get("invoices", [])
    ledger   = state.get("ledger_entries", [])
    if not invoices:
        return _r(_MIN, {}, "No invoices.")

    total = len(invoices)
    correct = 0
    fp = 0
    fraud_total = 0.0
    fraud_prevented = 0.0

    for inv in invoices:
        status = inv.get("status", "")
        is_f   = inv.get("is_fraudulent", False)
        is_d   = inv.get("is_duplicate", False)
        amt    = inv.get("amount", 0.0)
        inv_id = inv.get("invoice_id")

        if is_f:
            fraud_total += amt
            if status in ("flagged_fraud", "rejected"):
                correct += 1
                fraud_prevented += amt
            elif status == "posted":
                fp += 1
        elif is_d:
            if status in ("flagged_duplicate", "rejected"):
                correct += 1
            elif status == "posted":
                fp += 1
        else:
            if status == "posted" and any(e.get("invoice_id") == inv_id for e in ledger):
                correct += 1
            elif status in ("flagged_fraud", "flagged_duplicate"):
                fp += 1

    cls_score    = correct / total if total else 0.0
    prevent_rate = (fraud_prevented / fraud_total) if fraud_total > 0 else 0.95
    fp_penalty   = fp * 0.15

    raw = max(0.0, min(1.0, 0.7 * cls_score + 0.2 * prevent_rate - fp_penalty))
    score = _clamp(raw)
    bd = {"correct": correct, "total": total, "false_positives": fp,
          "cls_score": round(cls_score, 3), "prevent_rate": round(prevent_rate, 3)}
    fb = ("Excellent detection." if score >= 0.80
          else "Good, some misclassifications." if score >= 0.50
          else "Poor — many errors.")
    return _r(score, bd, fb)


# ── Hard ───────────────────────────────────────────────────────────────────

def _hard(state: dict) -> dict:
    invoices  = state.get("invoices", [])
    ledger    = state.get("ledger_entries", [])
    approvals = state.get("approvals_history", [])
    pos       = state.get("purchase_orders", [])
    if not invoices:
        return _r(_MIN, {}, "No invoices.")

    po_map = {p.get("po_number"): p for p in pos}
    total  = len(invoices)
    val_ok = appr_ok = ledger_ok = 0
    appr_need = ledger_need = 0

    for inv in invoices:
        status = inv.get("status", "")
        is_f   = inv.get("is_fraudulent", False)
        is_d   = inv.get("is_duplicate", False)
        amt    = inv.get("amount", 0.0)
        inv_id = inv.get("invoice_id")
        po_num = inv.get("po_number", "")

        if is_f and status in ("flagged_fraud", "rejected"):
            val_ok += 1
        elif is_d and status in ("flagged_duplicate", "rejected"):
            val_ok += 1
        elif not is_f and not is_d and status in (
                "validated", "approved", "awaiting_approval", "posted"):
            val_ok += 1

        if not is_f and not is_d and amt >= 7500.0:
            appr_need += 1
            po  = po_map.get(po_num)
            exp = po.get("approver_id", "") if po else ""
            rec = next((a for a in approvals if a.get("invoice_id") == inv_id), None)
            if rec and rec.get("approver_id") == exp and rec.get("decision") == "approved":
                appr_ok += 1

        if not is_f and not is_d:
            ledger_need += 1
            po     = po_map.get(po_num)
            exp_gl = po.get("gl_code", "") if po else ""
            entry  = next((e for e in ledger if e.get("invoice_id") == inv_id), None)
            if entry and status == "posted" and entry.get("gl_code") == exp_gl:
                ledger_ok += 1

    val_acc    = val_ok    / total       if total       else 0.0
    appr_acc   = appr_ok   / appr_need   if appr_need   else 0.50
    ledger_acc = ledger_ok / ledger_need if ledger_need else 0.0

    raw = max(0.0, min(1.0, 0.4 * val_acc + 0.3 * appr_acc + 0.3 * ledger_acc))
    score = _clamp(raw)
    bd = {
        "validation_accuracy":        round(val_acc,    3),
        "approval_routing_accuracy":  round(appr_acc,   3),
        "ledger_accuracy":            round(ledger_acc, 3),
        "appr_needed": appr_need, "appr_ok": appr_ok,
        "ledger_needed": ledger_need, "ledger_ok": ledger_ok,
    }
    fb = ("Strong AP cycle." if score >= 0.70
          else "Moderate — some steps missed." if score >= 0.40
          else "Weak — significant errors.")
    return _r(score, bd, fb)


# ── Helper ─────────────────────────────────────────────────────────────────

def _r(score: float, breakdown: dict, feedback: str) -> dict:
    safe_score = _clamp(score)
    return {
        "score":     safe_score,
        "breakdown": breakdown,
        "passed":    safe_score >= 0.50,
        "feedback":  feedback,
    }
