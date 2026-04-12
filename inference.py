"""
inference.py — Baseline inference for AP Workflow OpenEnv (OpenEnv stdout spec)

Usage:
    export OPENAI_API_KEY=sk-...
    python inference.py

STDOUT FORMAT:
    [START] task=<task_id> env=ap-workflow model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations
import json, os, sys, textwrap
from typing import Any, Dict, List, Optional
import requests
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.getenv("AP_ENV_URL", "http://localhost:7860")
BENCHMARK    = "ap-workflow"
MAX_STEPS    = 50
TEMPERATURE  = 0.1
MAX_TOKENS   = 400
TASKS = ["single_invoice_validation", "fraud_duplicate_detection", "full_ap_cycle"]

def log_start(task, model): print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)
def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)
def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

def env_reset(task_id, seed=42):
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status(); return r.json()

def env_step(tool_call, parameters, reasoning=""):
    r = requests.post(f"{ENV_URL}/step", json={"tool_call": tool_call, "parameters": parameters, "reasoning": reasoning}, timeout=30)
    r.raise_for_status(); return r.json()

def env_grade(task_id):
    r = requests.post(f"{ENV_URL}/grade", json={"task_id": task_id}, timeout=30)
    r.raise_for_status(); return r.json()

SYSTEM = textwrap.dedent("""
You are an expert AP clerk AI agent. Reply ONLY with JSON — no markdown:
{"tool_call": "<tool>", "parameters": {...}, "reasoning": "<brief>"}

Tools: validate_data | flag_fraud | mark_duplicate | route_approval | post_ledger | reject | request_info

Rules:
1. Always validate_data FIRST (unless clearly fraud/duplicate).
2. FAKE vendor name or round-number spike (10000/15000/etc) with no real PO → flag_fraud.
3. Identical to a previous invoice → mark_duplicate.
4. Amount > 7500 and valid → route_approval with approver_id from PO.
5. After validation+approval → post_ledger with gl_code from PO.
""").strip()

def get_action(client, obs, step, history):
    inv = obs.get("current_invoice")
    pos = obs.get("purchase_orders", [])
    pending = obs.get("pending_invoices", [])
    prompt = (f"Step {step} | Task: {obs.get('task_id')} | {obs.get('invoices_processed',0)}/{obs.get('total_invoices',0)} done\n\n"
              f"CURRENT INVOICE:\n{json.dumps(inv, indent=2, default=str) if inv else 'None'}\n\n"
              f"POs:\n{json.dumps(pos, indent=2, default=str)}\n\n"
              f"OTHER PENDING: {json.dumps([{'id':i.get('invoice_id'),'vendor':i.get('vendor_name'),'amount':i.get('amount')} for i in pending[1:]])}\n\n"
              f"LAST RESULT: {obs.get('last_action_result','None')}\n\nReply JSON:")
    msgs = [{"role": "system", "content": SYSTEM}] + history[-6:] + [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return "request_info", {"invoice_id": "unknown", "field": "all"}, str(e)
    try:
        p = json.loads(raw.replace("```json","").replace("```","").strip())
        return p.get("tool_call","request_info"), p.get("parameters",{}), p.get("reasoning","")
    except Exception:
        for t in ["validate_data","flag_fraud","mark_duplicate","route_approval","post_ledger","reject"]:
            if t in raw: return t, {}, "fallback"
        return "request_info", {}, "parse_error"

def run_task(client, task_id):
    log_start(task_id, MODEL_NAME)
    rewards: List[float] = []; steps_taken = 0; score = 0.0; success = False; history = []
    try:
        obs = env_reset(task_id)["observation"]
        for step in range(1, MAX_STEPS + 1):
            if not obs.get("pending_invoices") and step > 1: break
            tool, params, reason = get_action(client, obs, step, history)
            action_str = f"{tool}({json.dumps(params)})"
            try:
                resp = env_step(tool, params, reason)
                reward, done, error = resp.get("reward", 0.0), resp.get("done", False), None
                obs = resp.get("observation", obs)
                history += [{"role":"assistant","content":action_str}, {"role":"user","content":f"Result: {obs.get('last_action_result','')}"}]
            except Exception as e:
                reward, done, error = 0.0, False, str(e)
            rewards.append(reward); steps_taken = step
            log_step(step, action_str, reward, done, error)
            if done: break
        grade = env_grade(task_id)
        score, success = grade.get("score", 0.0), grade.get("passed", False)
        print(f"[DEBUG] breakdown: {grade.get('breakdown')} | {grade.get('feedback')}", flush=True)
    except Exception as e:
        print(f"[DEBUG] error: {e}", flush=True)
    finally:
        log_end(success, steps_taken, score, rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}

def main():
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY or HF_TOKEN.", file=sys.stderr); sys.exit(1)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[INFO] model={MODEL_NAME} env={ENV_URL}", flush=True)
    results = []
    for t in TASKS:
        print(f"\n{'='*55}\n[INFO] TASK: {t}\n{'='*55}", flush=True)
        results.append(run_task(client, t))
    print(f"\n{'='*55}\n[SUMMARY]", flush=True)
    for r in results:
        print(f"  {r['task_id']:42s} score={r['score']:.3f}  steps={r['steps']}", flush=True)
    print(f"\n  Average: {sum(r['score'] for r in results)/len(results):.3f}\n{'='*55}", flush=True)

if __name__ == "__main__":
    main()
