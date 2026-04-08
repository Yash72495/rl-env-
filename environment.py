"""
EmailTriageEnv – core OpenEnv environment.
Implements: reset() → observation, step(action) → (obs, reward, done, info), state()
"""

import uuid
import random
from typing import Optional, Dict, Any, List, Tuple

from app.models import EmailObservation, TriageAction, EnvState
from app.data import EMAILS, GT_LOOKUP
from app.tasks import TASKS, get_step_reward, grade_episode


class EmailTriageEnv:
    """
    Real-world email triage environment.

    An agent receives emails one-by-one and must:
    1. Classify the category (spam/urgent/normal/newsletter/internal)
    2. Assign a priority (1=critical … 5=ignore)
    3. Route to the correct department
    4. Decide whether a reply is needed

    Three tasks of increasing difficulty share this environment.
    """

    def __init__(self):
        self._episodes: Dict[str, Dict[str, Any]] = {}  # episode_id → state
        self._email_lookup = {e["email_id"]: e for e in EMAILS}

    # ── reset() ───────────────────────────────────────────────────────────────
    def reset(self, task_id: str = "task_spam_detection", seed: Optional[int] = None) -> Tuple[EmailObservation, str]:
        """
        Start a new episode.
        Returns (initial_observation, episode_id).
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}")

        task = TASKS[task_id]
        episode_id = str(uuid.uuid4())

        rng = random.Random(seed)
        email_ids = list(task["email_ids"])
        rng.shuffle(email_ids)
        # Use up to max_steps emails
        email_ids = email_ids[: task["max_steps"]]

        inbox = [self._email_lookup[eid] for eid in email_ids if eid in self._email_lookup]

        self._episodes[episode_id] = {
            "episode_id": episode_id,
            "task_id": task_id,
            "inbox": inbox,
            "current_step": 0,
            "max_steps": len(inbox),
            "total_reward": 0.0,
            "done": False,
            "log": [],  # list of {email_id, action, reward}
        }

        obs = self._make_observation(episode_id)
        return obs, episode_id

    # ── step() ────────────────────────────────────────────────────────────────
    def step(
        self, episode_id: str, action: Dict[str, Any]
    ) -> Tuple[Optional[EmailObservation], float, bool, Dict[str, Any]]:
        """
        Apply action to current email.
        Returns (next_observation | None, reward, done, info).
        """
        ep = self._get_episode(episode_id)

        if ep["done"]:
            return None, 0.0, True, {"error": "Episode already done. Call reset()."}

        step = ep["current_step"]
        current_email = ep["inbox"][step]
        email_id = current_email["email_id"]
        task_id = ep["task_id"]

        # Validate action types
        action = self._validate_action(action)

        # Compute reward
        reward, step_info = get_step_reward(action, email_id, task_id)

        # Log
        ep["log"].append({"email_id": email_id, "action": action, "reward": reward})
        ep["total_reward"] += reward
        ep["current_step"] += 1

        done = ep["current_step"] >= ep["max_steps"]
        ep["done"] = done

        info = {
            "step": step,
            "email_id": email_id,
            "task_id": task_id,
            **step_info,
        }

        if done:
            final = grade_episode(ep["log"], task_id)
            info["episode_score"] = final["score"]
            info["episode_breakdown"] = final["breakdown"]
            info["episode_summary"] = final["summary"]
            return None, reward, True, info

        next_obs = self._make_observation(episode_id)
        return next_obs, reward, False, info

    # ── state() ───────────────────────────────────────────────────────────────
    def state(self, episode_id: str) -> EnvState:
        """Return full current state of an episode."""
        ep = self._get_episode(episode_id)
        return EnvState(
            episode_id=episode_id,
            task_id=ep["task_id"],
            current_step=ep["current_step"],
            max_steps=ep["max_steps"],
            total_reward=ep["total_reward"],
            done=ep["done"],
            emails_triaged=len(ep["log"]),
            correct_decisions=sum(
                1 for entry in ep["log"] if entry["reward"] > 0
            ),
            inbox=[
                {k: v for k, v in e.items() if k not in ("gt_category", "gt_priority", "gt_department", "gt_should_reply")}
                for e in ep["inbox"]
            ],
        )

    # ── grader ────────────────────────────────────────────────────────────────
    def grade(self, episode_id: str) -> Dict[str, Any]:
        """Grade a completed (or in-progress) episode. Returns 0.0–1.0 score."""
        ep = self._get_episode(episode_id)
        return grade_episode(ep["log"], ep["task_id"])

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_episode(self, episode_id: str) -> Dict[str, Any]:
        if episode_id not in self._episodes:
            raise KeyError(f"Episode {episode_id} not found. Call /reset first.")
        return self._episodes[episode_id]

    def _make_observation(self, episode_id: str) -> EmailObservation:
        ep = self._episodes[episode_id]
        step = ep["current_step"]
        email = ep["inbox"][step]
        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            sender=email["sender"],
            body=email["body"],
            sender_domain=email["sender_domain"],
            has_attachment=email["has_attachment"],
            word_count=email["word_count"],
            inbox_position=step,
            current_step=step,
            max_steps=ep["max_steps"],
            emails_remaining=ep["max_steps"] - step,
            task_id=ep["task_id"],
        )

    def _validate_action(self, action: dict) -> dict:
        """Coerce types and fill defaults."""
        from app.data import VALID_CATEGORIES, VALID_DEPARTMENTS
        result = {
            "category": str(action.get("category", "normal")),
            "priority": int(action.get("priority", 3)),
            "department": str(action.get("department", "support")),
            "should_reply": bool(action.get("should_reply", False)),
        }
        # Clamp priority
        result["priority"] = max(1, min(5, result["priority"]))
        return result


# Singleton
_env = EmailTriageEnv()


def get_env() -> EmailTriageEnv:
    return _env
