#!/usr/bin/env python3
"""
Baseline inference script for EmailTriageEnv.

Uses the OpenAI API client to run an LLM agent against all 3 tasks.
Reads: OPENAI_API_KEY (env var), ENV_BASE_URL (default: http://localhost:7860)

Usage:
    python baseline.py
    ENV_BASE_URL=https://your-hf-space.hf.space python baseline.py

Produces a reproducible baseline score on all 3 tasks.
"""

import os
import json
import sys
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
SEED = 42  # Fixed seed for reproducibility

TASK_IDS = [
    "task_spam_detection",
    "task_priority_routing",
    "task_full_triage",
]

SYSTEM_PROMPT = """You are an expert email triage assistant. 
You will be shown one email at a time and must classify it.

For each email, respond with ONLY valid JSON in this exact format:
{
  "category": "<spam|urgent|normal|newsletter|internal>",
  "priority": <1-5>,
  "department": "<sales|support|engineering|hr|management|ignore>",
  "should_reply": <true|false>
}

Guidelines:
- category: spam=unwanted/phishing, urgent=needs immediate action, normal=regular business, 
            newsletter=subscriptions/digests, internal=from within company
- priority: 1=critical (fix now), 2=high, 3=medium, 4=low, 5=can ignore
- department: which team should handle it; use "ignore" for spam/newsletters
- should_reply: true if the email needs a response

Respond ONLY with JSON, no extra text."""


def call_env(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    url = f"{ENV_BASE_URL}{endpoint}"
    try:
        if method == "POST":
            r = requests.post(url, json=body, timeout=30)
        else:
            r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ API error calling {url}: {e}")
        sys.exit(1)


def parse_llm_action(response_text: str) -> dict:
    """Parse LLM response into action dict. Robust to minor formatting issues."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        action = json.loads(text)
        # Validate and clamp
        action["category"] = str(action.get("category", "normal"))
        action["priority"] = max(1, min(5, int(action.get("priority", 3))))
        action["department"] = str(action.get("department", "support"))
        action["should_reply"] = bool(action.get("should_reply", False))
        return action
    except (json.JSONDecodeError, ValueError, TypeError):
        print(f"  ⚠ Could not parse LLM response: {response_text[:100]}")
        return {"category": "normal", "priority": 3, "department": "support", "should_reply": False}


def run_task_with_llm(client: OpenAI, task_id: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Running task: {task_id}")
    print(f"{'='*60}")

    # Reset
    reset_resp = call_env("/reset", "POST", {"task_id": task_id, "seed": SEED})
    episode_id = reset_resp["episode_id"]
    obs = reset_resp["observation"]
    task_info = reset_resp["task"]

    print(f"  Episode ID : {episode_id}")
    print(f"  Task       : {task_info['name']} ({task_info['difficulty']})")
    print(f"  Max steps  : {obs['max_steps']}")
    print()

    step_num = 0
    total_reward = 0.0

    while True:
        # Build prompt for LLM
        user_msg = f"""Email #{step_num + 1}:
Subject: {obs['subject']}
From: {obs['sender']} ({obs['sender_domain']})
Has attachment: {obs['has_attachment']}
Word count: {obs['word_count']}

Body:
{obs['body']}

Classify this email."""

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,  # deterministic
                seed=SEED,
                max_tokens=150,
            )
            llm_text = completion.choices[0].message.content
        except Exception as e:
            print(f"  ⚠ LLM call failed: {e}. Using default action.")
            llm_text = '{"category": "normal", "priority": 3, "department": "support", "should_reply": false}'

        action = parse_llm_action(llm_text)

        # Step
        step_resp = call_env("/step", "POST", {"episode_id": episode_id, "action": action})
        reward = step_resp["reward"]
        done = step_resp["done"]
        info = step_resp["info"]
        total_reward += reward

        print(f"  Step {step_num + 1:2d}: email={obs['email_id']} | "
              f"action={action['category']}/{action['priority']}/{action['department']} | "
              f"reward={reward:+.2f}")

        step_num += 1

        if done:
            print(f"\n  Episode done after {step_num} steps.")
            print(f"  Total reward : {total_reward:.3f}")
            if "episode_score" in info:
                print(f"  Episode score: {info['episode_score']:.4f}")
                print(f"  Summary      : {info.get('episode_summary', '')}")
            break

        obs = step_resp["observation"]
        if obs is None:
            break

    # Get grader score
    grader_resp = call_env("/grader", "POST", {"episode_id": episode_id, "task_id": task_id})
    score = grader_resp["score"]
    breakdown = grader_resp["breakdown"]

    print(f"\n  ✓ Grader score: {score:.4f}")
    print(f"  Breakdown:")
    for k, v in breakdown.items():
        print(f"    {k}: {v}")

    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "steps": step_num,
        "total_reward": total_reward,
        "score": score,
        "breakdown": breakdown,
    }


def main():
    print("EmailTriageEnv – LLM Baseline Inference Script")
    print(f"Environment URL : {ENV_BASE_URL}")
    print(f"Model           : {MODEL}")
    print(f"Seed            : {SEED}")

    # Health check
    health = call_env("/health")
    print(f"Environment     : {health.get('environment', 'unknown')} v{health.get('version', '?')}")

    if not OPENAI_API_KEY:
        print("\n⚠ OPENAI_API_KEY not set. Using /baseline heuristic endpoint instead.\n")
        baseline_resp = call_env("/baseline", "POST")
        print("Baseline (heuristic) results:")
        for task_id, result in baseline_resp["scores"].items():
            print(f"  {task_id}: score={result['score']:.4f}")
        print(f"\nAgent: {baseline_resp['agent']}")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)

    all_results = []
    for task_id in TASK_IDS:
        result = run_task_with_llm(client, task_id)
        all_results.append(result)

    print(f"\n{'='*60}")
    print("FINAL BASELINE SCORES")
    print(f"{'='*60}")
    for r in all_results:
        difficulty = {"task_spam_detection": "easy", "task_priority_routing": "medium", "task_full_triage": "hard"}.get(r["task_id"], "?")
        print(f"  [{difficulty:6s}] {r['task_id']}: {r['score']:.4f}")

    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"\nBaseline run complete. Results are reproducible with seed={SEED}.")


if __name__ == "__main__":
    main()
