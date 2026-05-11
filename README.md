---
title: AP Workflow OpenEnv
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0.0"
python_version: "3.11"
app_file: server/app.py
pinned: false
---

# AP Workflow OpenEnv 🧾

[![OpenEnv](https://img.shields.io/badge/OpenEnv-spec__v1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

**AP Workflow OpenEnv** simulates the **Accounts Payable invoice processing workflow** — a critical real-world finance task. AP clerks handle incoming invoices by:

1. **Validating** invoice data against Purchase Orders (vendor, amount, date)
2. **Detecting** fraudulent or duplicate invoices before payment
3. **Routing** high-value invoices (>$7,500) for manager approval
4. **Posting** approved invoices to the correct General Ledger (GL) account

Humans spend **20–40 hours/week** on AP tasks. This environment trains and evaluates AI agents on this domain with dense reward shaping and deterministic graders.

---

## Project Structure

```
ap_workflow_env/
├── openenv.yaml          # OpenEnv manifest (spec_version: 1)
├── pyproject.toml        # Python package config
├── models.py             # APAction, APObservation (Pydantic)
├── client.py             # APEnv client (EnvClient)
├── data_generator.py     # Deterministic invoice/PO/fraud generation
├── grader.py             # Deterministic task graders (0.0–1.0)
├── inference.py          # Baseline script (OpenAI client, OpenEnv stdout)
├── test_env.py           # Smoke tests
├── Dockerfile            # Root Dockerfile
├── __init__.py           # Exports APAction, APObservation, APEnv
├── README.md             # This file
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI app (create_app or fallback)
    ├── ap_environment.py # APEnvironment logic
    ├── requirements.txt  # Server deps
    └── Dockerfile        # Server Dockerfile (openenv-base)
```

---

## Action Space

All actions are `APAction` objects with `tool_call`, `parameters`, and optional `reasoning`.

| Tool | Parameters | Description |
|------|-----------|-------------|
| `validate_data` | `invoice_id`, `amount`, `vendor_id` | Verify invoice vs PO |
| `flag_fraud` | `invoice_id`, `reason` | Mark as fraudulent |
| `mark_duplicate` | `invoice_id`, `original_id` | Mark as duplicate |
| `route_approval` | `invoice_id`, `approver_id`, `amount` | Route for manager approval |
| `post_ledger` | `invoice_id`, `gl_code`, `amount` | Post to General Ledger |
| `reject` | `invoice_id`, `reason` | Reject invalid invoice |
| `request_info` | `invoice_id`, `field` | Request more info |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `task_prompt` | str | Plain-English objective |
| `step_number` | int | Current step count |
| `current_invoice` | dict | Next invoice to process |
| `pending_invoices` | list | All unprocessed invoices |
| `purchase_orders` | list | Available POs for matching |
| `ledger_entries` | list | Posted ledger entries |
| `approvals_history` | list | Manager approval records |
| `last_action_result` | str | Feedback from previous action |
| `available_actions` | list | Valid tool names |
| `done` | bool | Episode complete flag |

---

## Tasks

### Easy: `single_invoice_validation`
- **Max Steps:** 20
- **Description:** Validate one invoice vs PO, post to correct GL account.
- **Grader:** `1.0` correct post → `0.7` wrong GL → `0.5` not posted → `0.0` rejected
- **Baseline (GPT-4o-mini):** ~0.92

### Medium: `fraud_duplicate_detection`
- **Max Steps:** 35
- **Description:** Process 3 invoices (1 valid, 1 fraud, 1 duplicate). Classify each correctly.
- **Grader:** `0.7 × classification_acc + 0.2 × fraud_prevented_ratio − fp_penalty`
- **Baseline (GPT-4o-mini):** ~0.67

### Hard: `full_ap_cycle`
- **Max Steps:** 50
- **Description:** Process 5 invoices with approval routing, fraud/dup detection, GL posting.
- **Grader:** `0.4 × val_acc + 0.3 × approval_acc + 0.3 × ledger_acc`
- **Baseline (GPT-4o-mini):** ~0.41

---

## Reward Shaping

| Event | Reward |
|-------|--------|
| Field validation (per field) | +0.10 |
| Fraud caught | +0.30 |
| Duplicate caught | +0.30 |
| Correct approval routing | +0.30 |
| Posted to correct GL | +0.50 |
| Posted to wrong GL | +0.20 |
| Valid invoice rejected | −0.30 |
| False positive fraud/dup | −0.10 |
| Hallucinated invoice ID | −0.20 |
| Loop penalty (step > 20) | −0.05/step |

---

## Setup & Usage

### Local

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t ap-env .
docker run -p 7860:7860 ap-env
```

### Run Baseline

```bash
export OPENAI_API_KEY=sk-your-key
export AP_ENV_URL=http://localhost:7860
python inference.py
```

---

## API Reference

### `POST /reset`
```json
{"task_id": "single_invoice_validation", "seed": 42}
```

### `POST /step`
```json
{"tool_call": "validate_data", "parameters": {"invoice_id": "INV-...", "amount": 1000}, "reasoning": "..."}
```

### `GET /state`
Returns full internal state snapshot.

### `POST /grade`
```json
{"task_id": "single_invoice_validation"}
```
Returns: `{"score": 0.92, "breakdown": {...}, "passed": true, "feedback": "..."}`

---

## Baseline Scores (GPT-4o-mini, seed=42)

| Task | Score | Steps |
|------|-------|-------|
| single_invoice_validation | ~0.92 | ~5 |
| fraud_duplicate_detection | ~0.67 | ~18 |
| full_ap_cycle | ~0.41 | ~42 |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | LLM API key |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model name |
| `AP_ENV_URL` | `http://localhost:7860` | Env server URL |
| `AP_ENV_SEED` | `42` | Random seed |
| `PORT` | `7860` | Server port |

---

## License

MIT — see [LICENSE](LICENSE) 

# Automated Accounts Payable System
An AI-powered solution designed to automate invoice processing and financial workflows.

## 🚀 Overview
This project was developed for the **Meta x Scaler Hackathon**. It simplifies the complex task of managing accounts payable by using Generative AI to extract data and automate validation.

## 🛠️ Tech Stack & AI Integration
- **Backend Logic & Architecture:** Developed using **Anthropic Claude**. Claude helped in structuring the complex financial logic and error handling.
- **Data Extraction & Processing:** Powered by **Google Gemini**. It accurately reads and interprets invoice details.
- **Research & Refinement:** Utilized **Perplexity AI** for real-time compliance checks.
- **Deployment:** Fully hosted on **Hugging Face Spaces**.

## ✨ Key Features
- **Intelligent Backend:** A robust system architecture built with Claude for high reliability.
- **Smart Data Extraction:** Uses Gemini to automate the manual entry of financial records.
- **End-to-End Automation:** Connects unstructured invoice data to a structured payable workflow.
- **Cloud Scale:** Deployed on Hugging Face for easy demonstration and scaling.

## 📸 Demo
[https://huggingface.co/spaces/GargiG15/Automated_Accounts_Payable/tree/main]
