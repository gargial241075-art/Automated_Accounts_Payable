"""test_env.py — Smoke tests. Run: python test_env.py"""
import sys, traceback

def ok(msg):   print(f"  PASS  {msg}", flush=True)
def fail(t, e): print(f"  FAIL  {t}: {e}", flush=True); traceback.print_exc()

def test_imports():
    from models import APAction, APObservation
    from server.ap_environment import APEnvironment
    from grader import grade_task
    from data_generator import build_easy, build_medium, build_hard
    ok("imports")

def test_reset_all():
    from server.ap_environment import APEnvironment
    env = APEnvironment(42)
    for tid in ["single_invoice_validation", "fraud_duplicate_detection", "full_ap_cycle"]:
        obs = env.reset(tid)
        assert obs.task_id == tid and obs.step_number == 0
        assert len(obs.pending_invoices) > 0
        ok(f"reset({tid}) — {len(obs.pending_invoices)} invoices")

def test_validate():
    from server.ap_environment import APEnvironment
    from models import APAction
    env = APEnvironment(42)
    obs = env.reset("single_invoice_validation")
    inv = obs.current_invoice
    o, reward, done, info = env.step(APAction(
        tool_call="validate_data",
        parameters={"invoice_id": inv["invoice_id"], "amount": inv["amount"], "vendor_id": inv["vendor_id"]}
    ))
    assert reward >= 0
    ok(f"validate_data reward={reward:.2f}")

def test_full_easy():
    from server.ap_environment import APEnvironment
    from models import APAction
    env = APEnvironment(42)
    obs = env.reset("single_invoice_validation")
    inv = obs.current_invoice
    po  = obs.purchase_orders[0] if obs.purchase_orders else None
    env.step(APAction(tool_call="validate_data",
        parameters={"invoice_id": inv["invoice_id"], "amount": inv["amount"], "vendor_id": inv["vendor_id"]}))
    o, reward, done, _ = env.step(APAction(tool_call="post_ledger",
        parameters={"invoice_id": inv["invoice_id"],
                    "gl_code": po["gl_code"] if po else "GL-5100",
                    "amount": inv["amount"]}))
    assert done, "Should be done after posting sole invoice"
    ok(f"full easy episode done=True reward={reward:.2f}")

def test_grader_easy():
    from server.ap_environment import APEnvironment
    from models import APAction
    from grader import grade_task
    env = APEnvironment(42)
    obs = env.reset("single_invoice_validation")
    inv = obs.current_invoice
    po  = obs.purchase_orders[0] if obs.purchase_orders else None
    env.step(APAction(tool_call="validate_data",
        parameters={"invoice_id": inv["invoice_id"], "amount": inv["amount"], "vendor_id": inv["vendor_id"]}))
    env.step(APAction(tool_call="post_ledger",
        parameters={"invoice_id": inv["invoice_id"],
                    "gl_code": po["gl_code"] if po else "GL-5100",
                    "amount": inv["amount"]}))
    result = grade_task(env.state(), "single_invoice_validation")
    assert 0.0 <= result["score"] <= 1.0
    ok(f"grader easy score={result['score']:.3f} passed={result['passed']}")

def test_grader_medium():
    from server.ap_environment import APEnvironment
    from models import APAction
    from grader import grade_task
    env = APEnvironment(42)
    obs = env.reset("fraud_duplicate_detection")
    for inv in list(obs.pending_invoices):
        if inv.get("is_fraudulent"):
            env.step(APAction(tool_call="flag_fraud", parameters={"invoice_id": inv["invoice_id"], "reason": "suspicious"}))
        elif inv.get("is_duplicate"):
            env.step(APAction(tool_call="mark_duplicate", parameters={"invoice_id": inv["invoice_id"], "original_id": ""}))
        else:
            po = next((p for p in obs.purchase_orders if p["vendor_id"] == inv["vendor_id"]), None)
            env.step(APAction(tool_call="validate_data", parameters={"invoice_id": inv["invoice_id"], "amount": inv["amount"], "vendor_id": inv["vendor_id"]}))
            env.step(APAction(tool_call="post_ledger", parameters={"invoice_id": inv["invoice_id"],
                "gl_code": po["gl_code"] if po else "GL-5100", "amount": inv["amount"]}))
    result = grade_task(env.state(), "fraud_duplicate_detection")
    assert 0.0 <= result["score"] <= 1.0
    ok(f"grader medium score={result['score']:.3f}")

def test_state_keys():
    from server.ap_environment import APEnvironment
    env = APEnvironment(42)
    env.reset("full_ap_cycle")
    s = env.state()
    for k in ["task_id","step_count","invoices","purchase_orders","ledger_entries","approvals_history"]:
        assert k in s, f"Missing: {k}"
    ok("state() keys OK")

def test_hallucination_penalty():
    from server.ap_environment import APEnvironment
    from models import APAction
    env = APEnvironment(42)
    env.reset("single_invoice_validation")
    _, reward, _, _ = env.step(APAction(tool_call="validate_data", parameters={"invoice_id": "INV-FAKEID"}))
    assert reward < 0
    ok(f"hallucination penalty reward={reward:.2f}")

def test_reset_clean():
    from server.ap_environment import APEnvironment
    from models import APAction
    env = APEnvironment(42)
    obs = env.reset("single_invoice_validation")
    inv = obs.current_invoice
    env.step(APAction(tool_call="validate_data", parameters={"invoice_id": inv["invoice_id"], "amount": inv["amount"], "vendor_id": inv["vendor_id"]}))
    obs2 = env.reset("single_invoice_validation")
    assert obs2.step_number == 0 and obs2.ledger_entries == []
    ok("reset() produces clean state")

TESTS = [test_imports, test_reset_all, test_validate, test_full_easy,
         test_grader_easy, test_grader_medium, test_state_keys,
         test_hallucination_penalty, test_reset_clean]

if __name__ == "__main__":
    passed = failed = 0
    print("="*50 + "\nAP Workflow OpenEnv — Tests\n" + "="*50)
    for t in TESTS:
        try: t(); passed += 1
        except Exception as e: fail(t.__name__, e); failed += 1
    print("="*50)
    print(f"Result: {passed} passed / {failed} failed")
    print("="*50)
    sys.exit(0 if failed == 0 else 1)
