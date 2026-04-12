"""data_generator.py — Deterministic invoice/PO generation. stdlib only."""
from __future__ import annotations
import random
import uuid
from datetime import date, timedelta

VENDORS = [
    ("Acme Supplies Co.",       "V001", "GL-5100", "Operations",  "MGR001"),
    ("TechParts Ltd.",          "V002", "GL-5200", "Engineering", "MGR002"),
    ("Global Office Solutions", "V003", "GL-5300", "Admin",       "MGR003"),
    ("FastShip Logistics",      "V004", "GL-5400", "Logistics",   "MGR004"),
    ("Pinnacle IT Services",    "V005", "GL-5200", "Engineering", "MGR002"),
    ("BlueSky Marketing",       "V006", "GL-5500", "Marketing",   "MGR005"),
]
HIGH_VALUE_THRESHOLD = 7500.0


def _uid() -> str:
    return uuid.uuid4().hex[:8].upper()


def _dt(base: date, offset: int = 0) -> str:
    return (base - timedelta(days=offset)).isoformat()


def _inv(invoice_id, vendor_name, vendor_id, amount, inv_date, due_date,
         po_number, is_fraud=False, is_dup=False, signals=None) -> dict:
    return {
        "invoice_id": invoice_id,
        "vendor_name": vendor_name,
        "vendor_id": vendor_id,
        "amount": amount,
        "date": inv_date,
        "due_date": due_date,
        "po_number": po_number,
        "line_items": [{"description": "Services", "quantity": 1,
                        "unit_price": amount, "total": amount}],
        "currency": "USD",
        "status": "pending",
        "is_duplicate": is_dup,
        "is_fraudulent": is_fraud,
        "fraud_signals": signals or [],
    }


def _po(po_number, vendor_id, approved_amount, approved_date,
        gl_code, dept, approver) -> dict:
    return {
        "po_number": po_number,
        "vendor_id": vendor_id,
        "approved_amount": approved_amount,
        "approved_date": approved_date,
        "gl_code": gl_code,
        "department": dept,
        "approver_id": approver,
    }


def make_valid(r: random.Random, base: date):
    name, vid, gl, dept, mgr = r.choice(VENDORS)
    amt = round(r.uniform(300.0, 5000.0), 2)
    d = _dt(base, r.randint(0, 10))
    po_num = f"PO-{_uid()}"
    invoice = _inv(f"INV-{_uid()}", name, vid, amt, d, _dt(base, -30), po_num)
    purchase_order = _po(po_num, vid, round(amt * r.uniform(0.98, 1.02), 2), d, gl, dept, mgr)
    return invoice, purchase_order


def make_fraud(r: random.Random, base: date):
    _, vid, gl, _, _ = r.choice(VENDORS)
    amt = float(r.choice([10000, 15000, 22500, 30000]))
    d = _dt(base, r.randint(0, 5))
    return _inv(
        f"INV-{_uid()}", f"FAKE-VENDOR-{_uid()[:4]}", vid,
        amt, d, _dt(base, -30), f"PO-FAKE-{_uid()}",
        is_fraud=True,
        signals=["unknown_vendor", "round_amount_spike", "no_matching_po"],
    )


def make_dup(original: dict) -> dict:
    dup = dict(original)
    dup["invoice_id"] = f"INV-{_uid()}"
    dup["is_duplicate"] = True
    return dup


def make_high_value(r: random.Random, base: date):
    name, vid, gl, dept, mgr = r.choice(VENDORS)
    amt = round(r.uniform(8000.0, 14000.0), 2)
    d = _dt(base, r.randint(0, 5))
    po_num = f"PO-{_uid()}"
    invoice = _inv(f"INV-{_uid()}", name, vid, amt, d, _dt(base, -30), po_num)
    purchase_order = _po(po_num, vid, amt, d, gl, dept, mgr)
    return invoice, purchase_order


def build_easy(seed: int = 42) -> dict:
    r = random.Random(seed)
    base = date(2025, 3, 15)
    inv, po = make_valid(r, base)
    return {"invoices": [inv], "purchase_orders": [po]}


def build_medium(seed: int = 42) -> dict:
    r = random.Random(seed)
    base = date(2025, 3, 15)
    inv1, po1 = make_valid(r, base)
    fraud = make_fraud(r, base)
    dup = make_dup(inv1)
    return {"invoices": [inv1, fraud, dup], "purchase_orders": [po1]}


def build_hard(seed: int = 42) -> dict:
    r = random.Random(seed)
    base = date(2025, 3, 15)
    inv1, po1 = make_valid(r, base)
    inv2, po2 = make_valid(r, base)
    inv3, po3 = make_high_value(r, base)
    fraud = make_fraud(r, base)
    dup = make_dup(inv1)
    invoices = [inv1, inv2, inv3, fraud, dup]
    r.shuffle(invoices)
    return {"invoices": invoices, "purchase_orders": [po1, po2, po3]}
