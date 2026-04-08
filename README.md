# 📧 EmailTriageEnv — OpenEnv Environment

> **A real-world email triage environment for training and evaluating AI agents.**

An AI agent processes a corporate inbox one email at a time, learning to classify, prioritize, route to departments, and decide whether a reply is needed — exactly the task performed by executive assistants and operations teams daily.

---

## 🎯 Motivation

Email triage is a high-value, high-volume task in every organization. It requires nuanced judgment: recognizing phishing, distinguishing "urgent" from "can wait", routing to the right team, and knowing when to reply. This environment captures that complexity in a structured, learnable format with clear success criteria.

---

## 🗺️ Environment Overview

| Property | Value |
|---|---|
| **Interface** | HTTP REST API (OpenEnv spec) |
| **Episode type** | Sequential email processing |
| **Observation** | Email content + metadata |
| **Action space** | category × priority × department × should_reply |
| **Reward range** | −1.0 to +1.0 per step |
| **Tasks** | 3 (easy → medium → hard) |

---

## 📦 Action & Observation Space

### Observation (what the agent sees)
```json
{
  "email_id": "e004",
  "subject": "URGENT: Production database is DOWN",
  "sender": "alerts@monitoring.company.com",
  "sender_domain": "company.com",
  "body": "CRITICAL ALERT: Production database...",
  "has_attachment": false,
  "word_count": 30,
  "inbox_position": 0,
  "current_step": 0,
  "max_steps": 10,
  "emails_remaining": 10,
  "task_id": "task_spam_detection"
}
```

### Action (what the agent does)
```json
{
  "category": "urgent",
  "priority": 1,
  "department": "engineering",
  "should_reply": true
}
```

**category**: `spam` | `urgent` | `normal` | `newsletter` | `internal`  
**priority**: `1` (critical) → `5` (ignore)  
**department**: `sales` | `support` | `engineering` | `hr` | `management` | `ignore`  
**should_reply**: `true` | `false`

---

## 🎮 Three Tasks

### Task 1: Spam Detection (Easy)
- **Goal**: Classify each email as spam or not-spam
- **Emails**: 10 (mix of spam, legitimate, urgent)
- **Scoring**: Spam F1 score (70%) + step reward signal (30%)
- **Success threshold**: 0.70
- **Key challenge**: Phishing emails designed to look legitimate

### Task 2: Priority & Department Routing (Medium)
- **Goal**: Assign correct priority (1–5) and route to right department
- **Emails**: 12 (business emails across all categories)
- **Scoring**: Priority accuracy ±1 (40%) + department accuracy (40%) + category (20%)
- **Success threshold**: 0.60
- **Key challenge**: Partial credit for close-but-not-exact priorities; adversarial routing

### Task 3: Full Email Triage (Hard)
- **Goal**: All 4 dimensions: category + priority + department + reply decision
- **Emails**: All 20 (full inbox simulation)
- **Scoring**: Weighted composite across all dimensions
- **Success threshold**: 0.55
- **Key challenge**: Adversarial emails (casual-tone urgencies, phishing that looks legit), simultaneous optimization

---

## 🏆 Reward Function

The reward is **shaped** — not just binary end-of-episode:

- **Per step**: +1.0 (perfect) → −1.0 (destructive decision)
- **Priority partial credit**: off by 1 = 0.5 reward, off by 2 = 0.25
- **Bonus**: Correctly routing critical incidents (+0.2), catching phishing (+0.1)
- **Penalty**: Marking urgent emails as spam (−1.0), invalid action values (−0.5)
- **Episode score**: 0.0–1.0 via grader, computed from full trajectory

---

## 🚀 Usage

### Start the server
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Interact with the API
```python
import requests

BASE = "http://localhost:7860"

# 1. Start episode
resp = requests.post(f"{BASE}/reset", json={"task_id": "task_spam_detection", "seed": 42})
episode_id = resp.json()["episode_id"]
obs = resp.json()["observation"]

# 2. Step through emails
while True:
    action = {
        "category": "spam" if "prize" in obs["subject"].lower() else "normal",
        "priority": 5 if "prize" in obs["subject"].lower() else 3,
        "department": "ignore" if "prize" in obs["subject"].lower() else "support",
        "should_reply": False
    }
    step = requests.post(f"{BASE}/step", json={"episode_id": episode_id, "action": action})
    result = step.json()
    if result["done"]:
        break
    obs = result["observation"]

# 3. Get score
score = requests.post(f"{BASE}/grader", json={"episode_id": episode_id, "task_id": "task_spam_detection"})
print(score.json()["score"])  # 0.0–1.0
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode, returns first observation |
| `/step` | POST | Submit action, receive reward + next obs |
| `/state/{id}` | GET | Current episode state |
| `/tasks` | GET | List tasks + action schemas |
| `/grader` | POST | Grade completed episode (0.0–1.0) |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/health` | GET | Health check |

---

## 🧪 Baseline Scores

Run the baseline:
```bash
# LLM-based (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... ENV_BASE_URL=http://localhost:7860 python baseline.py

# Heuristic fallback (no API key needed)
python baseline.py
```

Expected heuristic baseline scores:

| Task | Difficulty | Baseline Score |
|---|---|---|
| Spam Detection | Easy | ~0.72 |
| Priority Routing | Medium | ~0.51 |
| Full Triage | Hard | ~0.44 |

A strong LLM agent (GPT-4o) should achieve ~0.85 / 0.70 / 0.62.

---

## 📁 Project Structure

```
email-triage-env/
├── Dockerfile
├── openenv.yaml          # OpenEnv metadata
├── requirements.txt
├── README.md
├── baseline.py           # LLM inference script
└── app/
    ├── __init__.py
    ├── main.py           # FastAPI app + all endpoints
    ├── environment.py    # Core env: reset/step/state
    ├── models.py         # Pydantic typed models
    ├── tasks.py          # 3 tasks + programmatic graders
    └── data.py           # 20 synthetic emails w/ ground truth
```

---

## 🔧 Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

Interactive docs: http://localhost:7860/docs

---

## 📜 License

MIT
