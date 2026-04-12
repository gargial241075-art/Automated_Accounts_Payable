"""
server/app.py — FastAPI server for AP Workflow OpenEnv.

IMPORTANT: Zero optional imports at module level.
Every import is guaranteed to succeed when PYTHONPATH=/app (Docker).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Absolute imports — work when PYTHONPATH=/app (Docker CMD)
from models import APAction, APObservation          # noqa: E402
from server.ap_environment import APEnvironment     # noqa: E402
from grader import grade_task                       # noqa: E402

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AP Workflow OpenEnv",
    description="Automated Accounts Payable invoice processing environment.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = APEnvironment(seed=int(os.getenv("AP_ENV_SEED", "42")))

# ── Request schemas ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "single_invoice_validation"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    tool_call: str
    parameters: Dict[str, Any] = {}
    reasoning: Optional[str] = None


class GradeRequest(BaseModel):
    task_id: str = "single_invoice_validation"


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":    "AP Workflow OpenEnv",
        "version": "1.0.0",
        "tasks":   [
            "single_invoice_validation",
            "fraud_duplicate_detection",
            "full_ap_cycle",
        ],
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
    }


@app.get("/health")
async def health():
    """Health check — must return 200 for HF Space to be considered live."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment and return the initial observation.
    Returns HTTP 200 with the initial APObservation.
    """
    global _env
    try:
        seed = req.seed if req.seed is not None else int(os.getenv("AP_ENV_SEED", "42"))
        _env = APEnvironment(seed=seed)
        obs  = _env.reset(task_id=req.task_id, seed=seed)
        return {
            "observation": obs.model_dump(),
            "task_id":     req.task_id,
            "message":     f"Environment reset for task: {req.task_id}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {e}")


@app.post("/step")
async def step(req: StepRequest):
    """Execute one action. Returns observation, reward, done, info."""
    try:
        action = APAction(
            tool_call=req.tool_call,
            parameters=req.parameters,
            reasoning=req.reasoning,
        )
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward,
            "done":        done,
            "info":        info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {e}")


@app.get("/state")
async def state():
    """Return full internal environment state snapshot."""
    return _env.state()


@app.post("/grade")
async def grade(req: GradeRequest = GradeRequest()):
    """Grade the current episode. Returns score 0.0–1.0 with breakdown."""
    try:
        snap   = _env.state()
        result = grade_task(snap, req.task_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grade error: {e}")


@app.get("/tasks")
async def tasks():
    """List all available tasks."""
    return {"tasks": [
        {"id": "single_invoice_validation", "difficulty": "easy",   "max_steps": 20},
        {"id": "fraud_duplicate_detection", "difficulty": "medium", "max_steps": 35},
        {"id": "full_ap_cycle",             "difficulty": "hard",   "max_steps": 50},
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


def main():
    """Entry point for [project.scripts] serve command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
