"""
3 Tasks with programmatic graders (scores 0.0 – 1.0).

Task 1 (Easy)   – Spam Detection:    Classify spam vs non-spam
Task 2 (Medium) – Priority Routing:  Assign correct priority + department
Task 3 (Hard)   – Full Triage:       Category + priority + department + reply decision
"""

from typing import Dict, Any, List
from app.data import EMAILS, VALID_CATEGORIES, VALID_DEPARTMENTS

# ── Task definitions ──────────────────────────────────────────────────────────
TASKS = {
    "task_spam_detection": {
        "task_id": "task_spam_detection",
        "name": "Spam Detection",
        "description": (
            "Classify each incoming email as 'spam' or 'not_spam'. "
            "A binary classification task that tests the agent's ability to "
            "recognize phishing, unsolicited promotions, and malicious emails."
        ),
        "difficulty": "easy",
        "max_steps": 10,
        "email_ids": ["e001", "e002", "e003", "e008", "e009", "e013", "e014", "e015", "e004", "e019"],
        "success_threshold": 0.7,
        "action_schema": {
            "category": {
                "type": "string",
                "enum": ["spam", "urgent", "normal", "newsletter", "internal"],
                "description": "Classify the email. Only 'spam' vs anything else matters for scoring."
            },
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            "department": {"type": "string", "enum": list(VALID_DEPARTMENTS)},
            "should_reply": {"type": "boolean"},
        },
    },
    "task_priority_routing": {
        "task_id": "task_priority_routing",
        "name": "Priority & Department Routing",
        "description": (
            "Assign the correct priority (1-5) and route emails to the correct "
            "department. Tests understanding of business urgency and org structure. "
            "Partial credit for getting priority range correct."
        ),
        "difficulty": "medium",
        "max_steps": 12,
        "email_ids": ["e004", "e005", "e006", "e007", "e008", "e009", "e010", "e011", "e015", "e016", "e018", "e019"],
        "success_threshold": 0.6,
        "action_schema": {
            "category": {"type": "string", "enum": list(VALID_CATEGORIES)},
            "priority": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1=Critical, 2=High, 3=Medium, 4=Low, 5=Ignore"
            },
            "department": {
                "type": "string",
                "enum": list(VALID_DEPARTMENTS),
                "description": "Which team should handle this email"
            },
            "should_reply": {"type": "boolean"},
        },
    },
    "task_full_triage": {
        "task_id": "task_full_triage",
        "name": "Full Email Triage",
        "description": (
            "Complete end-to-end email triage: correctly classify category, "
            "assign priority, route to department, and decide if a reply is needed. "
            "Tests all dimensions simultaneously. Includes adversarial emails "
            "designed to confuse agents (phishing that looks legitimate, urgent "
            "emails with casual tone, etc.)."
        ),
        "difficulty": "hard",
        "max_steps": 20,
        "email_ids": [e["email_id"] for e in EMAILS],  # all 20 emails
        "success_threshold": 0.55,
        "action_schema": {
            "category": {"type": "string", "enum": list(VALID_CATEGORIES)},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            "department": {"type": "string", "enum": list(VALID_DEPARTMENTS)},
            "should_reply": {"type": "boolean"},
        },
    },
}

# ── Ground-truth lookup ───────────────────────────────────────────────────────
GT_LOOKUP: Dict[str, Dict] = {e["email_id"]: e for e in EMAILS}


def get_step_reward(action: dict, email_id: str, task_id: str) -> tuple[float, dict]:
    """
    Compute per-step reward for a given action on an email.
    Returns (reward, info_dict).
    """
    gt = GT_LOOKUP.get(email_id)
    if gt is None:
        return 0.0, {"error": "unknown email_id"}

    info = {}
    reward = 0.0

    cat_correct = action.get("category") == gt["gt_category"]
    pri = action.get("priority", 3)
    gt_pri = gt["gt_priority"]
    dept_correct = action.get("department") == gt["gt_department"]
    reply_correct = action.get("should_reply") == gt["gt_should_reply"]

    # Priority partial credit: exact=full, off by 1=half, off by 2=quarter, else=0
    pri_diff = abs(pri - gt_pri)
    if pri_diff == 0:
        pri_reward = 1.0
    elif pri_diff == 1:
        pri_reward = 0.5
    elif pri_diff == 2:
        pri_reward = 0.25
    else:
        pri_reward = 0.0

    if task_id == "task_spam_detection":
        # Only care about spam vs not-spam
        predicted_spam = action.get("category") == "spam"
        actual_spam = gt["gt_category"] == "spam"
        spam_correct = predicted_spam == actual_spam
        reward = 1.0 if spam_correct else -0.5
        # Penalty for labelling legitimate urgent emails as spam
        if not actual_spam and predicted_spam and gt["gt_category"] == "urgent":
            reward = -1.0  # heavy penalty
        info = {
            "spam_correct": spam_correct,
            "predicted_spam": predicted_spam,
            "actual_spam": actual_spam,
        }

    elif task_id == "task_priority_routing":
        # 50% priority, 50% department
        reward = 0.5 * pri_reward + 0.5 * (1.0 if dept_correct else -0.3)
        # Bonus for catching urgent emails as priority 1
        if gt["gt_category"] == "urgent" and gt_pri == 1 and pri == 1:
            reward += 0.2
        info = {
            "priority_reward": pri_reward,
            "dept_correct": dept_correct,
            "gt_priority": gt_pri,
            "given_priority": pri,
        }

    else:  # full triage
        # Weighted: category 35%, priority 25%, dept 25%, reply 15%
        reward = (
            0.35 * (1.0 if cat_correct else -0.3)
            + 0.25 * pri_reward
            + 0.25 * (1.0 if dept_correct else -0.3)
            + 0.15 * (1.0 if reply_correct else -0.2)
        )
        # Bonus for correctly handling adversarial phishing (looks legit but is spam)
        if gt["gt_category"] == "spam" and action.get("department") == "ignore":
            reward += 0.1
        info = {
            "cat_correct": cat_correct,
            "priority_reward": pri_reward,
            "dept_correct": dept_correct,
            "reply_correct": reply_correct,
        }

    # Global penalties
    if action.get("category") not in VALID_CATEGORIES:
        reward -= 0.5
        info["invalid_category_penalty"] = True
    if action.get("department") not in VALID_DEPARTMENTS:
        reward -= 0.5
        info["invalid_department_penalty"] = True

    reward = max(-1.0, min(1.0, reward))
    info["reward"] = reward
    return reward, info


def grade_episode(episode_log: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    """
    Compute final 0.0–1.0 score for a completed episode.
    episode_log: list of {"email_id": ..., "action": {...}, "reward": float}
    """
    if not episode_log:
        return {"score": 0.0, "breakdown": {}, "summary": "No actions taken"}

    task = TASKS.get(task_id)
    if not task:
        return {"score": 0.0, "breakdown": {}, "summary": "Unknown task"}

    total_emails = len(episode_log)
    correct_cats = 0
    correct_pris = 0
    correct_depts = 0
    correct_replies = 0
    spam_tp = spam_fp = spam_fn = spam_tn = 0
    sum_rewards = 0.0

    for entry in episode_log:
        eid = entry["email_id"]
        action = entry["action"]
        gt = GT_LOOKUP.get(eid, {})
        if not gt:
            continue

        if action.get("category") == gt.get("gt_category"):
            correct_cats += 1
        if abs(action.get("priority", 3) - gt.get("gt_priority", 3)) <= 1:
            correct_pris += 1
        if action.get("department") == gt.get("gt_department"):
            correct_depts += 1
        if action.get("should_reply") == gt.get("gt_should_reply"):
            correct_replies += 1
        sum_rewards += entry.get("reward", 0.0)

        # Spam metrics
        pred_spam = action.get("category") == "spam"
        actual_spam = gt.get("gt_category") == "spam"
        if pred_spam and actual_spam:
            spam_tp += 1
        elif pred_spam and not actual_spam:
            spam_fp += 1
        elif not pred_spam and actual_spam:
            spam_fn += 1
        else:
            spam_tn += 1

    cat_acc = correct_cats / total_emails
    pri_acc = correct_pris / total_emails
    dept_acc = correct_depts / total_emails
    reply_acc = correct_replies / total_emails

    # F1 for spam
    spam_precision = spam_tp / (spam_tp + spam_fp) if (spam_tp + spam_fp) > 0 else 0
    spam_recall = spam_tp / (spam_tp + spam_fn) if (spam_tp + spam_fn) > 0 else 0
    spam_f1 = (
        2 * spam_precision * spam_recall / (spam_precision + spam_recall)
        if (spam_precision + spam_recall) > 0
        else 0.0
    )

    if task_id == "task_spam_detection":
        score = 0.7 * spam_f1 + 0.3 * (max(0, sum_rewards) / total_emails)
    elif task_id == "task_priority_routing":
        score = 0.4 * pri_acc + 0.4 * dept_acc + 0.2 * cat_acc
    else:  # full triage
        score = 0.3 * cat_acc + 0.25 * pri_acc + 0.25 * dept_acc + 0.2 * reply_acc

    score = round(min(1.0, max(0.0, score)), 4)

    breakdown = {
        "category_accuracy": round(cat_acc, 3),
        "priority_within_1_accuracy": round(pri_acc, 3),
        "department_accuracy": round(dept_acc, 3),
        "reply_decision_accuracy": round(reply_acc, 3),
        "spam_f1": round(spam_f1, 3),
        "avg_step_reward": round(sum_rewards / total_emails, 3),
    }

    summary = (
        f"Score: {score:.3f} | Category acc: {cat_acc:.0%} | "
        f"Dept acc: {dept_acc:.0%} | Priority acc (±1): {pri_acc:.0%}"
    )

    return {"score": score, "breakdown": breakdown, "summary": summary}
