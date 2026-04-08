"""
FastAPI application for EmailTriageEnv.

Endpoints:
  POST /reset          – start new episode
  POST /step           – take action
  GET  /state/{id}     – current episode state
  GET  /tasks          – list all tasks + action schemas
  POST /grader         – grade a completed episode (0.0–1.0)
  POST /baseline       – run baseline agent on all 3 tasks, return scores
  GET  /health         – health check
"""

import os
import random
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment import get_env
from app.models import (
    EmailObservation, TriageAction, StepResult, EnvState,
    TaskInfo, GraderRequest, GraderResult
)
from app.tasks import TASKS

app = FastAPI(
    title="EmailTriageEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage emails: "
        "classify, prioritize, route to departments, and decide on replies. "
        "Three tasks of increasing difficulty (easy→medium→hard)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = get_env()


# ── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_spam_detection"
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    episode_id: str
    observation: EmailObservation
    task: dict


class StepRequest(BaseModel):
    episode_id: str
    action: TriageAction


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "EmailTriageEnv", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """Start a new episode. Returns first observation + episode_id."""
    try:
        obs, episode_id = env.reset(task_id=req.task_id, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResetResponse(
        episode_id=episode_id,
        observation=obs,
        task=TASKS[req.task_id],
    )


@app.post("/step")
def step(req: StepRequest):
    """
    Apply action to current email.
    Returns next observation (or null if done), reward, done, info.
    """
    try:
        action_dict = req.action.model_dump()
        obs, reward, done, info = env.step(req.episode_id, action_dict)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump() if obs else None,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state/{episode_id}", response_model=EnvState)
def state(episode_id: str):
    """Return current state of an episode."""
    try:
        return env.state(episode_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """Return all available tasks and their action schemas."""
    return {"tasks": list(TASKS.values())}


@app.post("/grader")
def grader(req: GraderRequest):
    """Grade a completed episode. Returns score 0.0–1.0."""
    try:
        result = env.grade(req.episode_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return GraderResult(
        episode_id=req.episode_id,
        task_id=req.task_id,
        score=result["score"],
        breakdown=result["breakdown"],
        summary=result["summary"],
    )


@app.post("/baseline")
def baseline():
    """
    Run a deterministic baseline agent on all 3 tasks.
    Uses a simple heuristic (keyword matching) – not LLM-based.
    Returns scores for each task.
    """
    results = {}
    for task_id in TASKS:
        score = _run_heuristic_baseline(task_id)
        results[task_id] = score

    return {
        "agent": "keyword_heuristic_baseline",
        "scores": results,
        "note": "Baseline uses simple keyword matching, not a language model.",
    }


# ── Heuristic baseline agent ──────────────────────────────────────────────────

SPAM_KEYWORDS = {"win", "free", "click", "prize", "discount", "offer", "verify", "urgent", "limited", "claim", "congratulations"}
URGENT_KEYWORDS = {"critical", "down", "breach", "incident", "immediately", "emergency", "outage", "corrupted", "p0"}
ENGINEERING_KEYWORDS = {"server", "database", "bug", "code", "api", "deploy", "memory", "production", "crash", "script"}
SALES_KEYWORDS = {"pricing", "partnership", "deal", "contract", "business", "client", "revenue"}
SUPPORT_KEYWORDS = {"refund", "complaint", "help", "issue", "broken", "unsubscribe", "frustrated", "disappointed"}
HR_KEYWORDS = {"payroll", "onboarding", "hire", "employee", "benefits", "hr", "policy"}
MANAGEMENT_KEYWORDS = {"meeting", "planning", "ceo", "executive", "invoice", "legal", "signature"}


def _classify_heuristic(subject: str, body: str, sender_domain: str):
    text = (subject + " " + body).lower()
    words = set(text.split())

    # Spam signals: suspicious domain OR spam keywords
    suspicious_domain = not any(d in sender_domain for d in ["company.com", "gmail.com", "outlook.com", "hotmail.com", "bigclient.com", "techcorp.com", "vendor-supplies.com", "startup.io"])
    spam_score = len(words & SPAM_KEYWORDS) + (3 if suspicious_domain else 0)

    if spam_score >= 3:
        return "spam", 5, "ignore", False

    urgent_score = len(words & URGENT_KEYWORDS)
    if urgent_score >= 2:
        dept = "engineering"
        if any(k in text for k in ["contract", "legal", "ceo"]):
            dept = "management"
        return "urgent", 1, dept, True

    # Department routing
    if len(words & ENGINEERING_KEYWORDS) > len(words & SALES_KEYWORDS):
        dept = "engineering"
    elif len(words & SALES_KEYWORDS) > 0:
        dept = "sales"
    elif len(words & SUPPORT_KEYWORDS) > 0:
        dept = "support"
    elif len(words & HR_KEYWORDS) > 0:
        dept = "hr"
    elif len(words & MANAGEMENT_KEYWORDS) > 0:
        dept = "management"
    else:
        dept = "support"

    # Newsletter detection
    if "unsubscribe" in text or "newsletter" in text or "digest" in text:
        return "newsletter", 5, "ignore", False

    # Internal detection
    if sender_domain == "company.com":
        return "internal", 3, dept, False

    priority = 3
    if urgent_score == 1:
        priority = 2
    should_reply = dept in {"sales", "support", "engineering"}

    return "normal", priority, dept, should_reply


def _run_heuristic_baseline(task_id: str) -> dict:
    from app.data import EMAILS
    from app.tasks import TASKS, get_step_reward, grade_episode

    task = TASKS[task_id]
    email_ids = task["email_ids"]
    email_lookup = {e["email_id"]: e for e in EMAILS}

    log = []
    for eid in email_ids:
        email = email_lookup.get(eid)
        if not email:
            continue
        cat, pri, dept, reply = _classify_heuristic(
            email["subject"], email["body"], email["sender_domain"]
        )
        action = {"category": cat, "priority": pri, "department": dept, "should_reply": reply}
        reward, _ = get_step_reward(action, eid, task_id)
        log.append({"email_id": eid, "action": action, "reward": reward})

    result = grade_episode(log, task_id)
    return result
