from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# ── Observation ──────────────────────────────────────────────────────────────
class EmailObservation(BaseModel):
    """What the agent sees at each step."""
    email_id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    body: str = Field(..., description="Email body text")
    sender_domain: str = Field(..., description="Domain of the sender")
    has_attachment: bool = Field(..., description="Whether email has attachments")
    word_count: int = Field(..., description="Word count of email body")
    inbox_position: int = Field(..., description="Position in inbox (0-indexed)")
    current_step: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps per episode")
    emails_remaining: int = Field(..., description="Emails left to triage")
    task_id: str = Field(..., description="Current task identifier")


# ── Action ────────────────────────────────────────────────────────────────────
class TriageAction(BaseModel):
    """Action the agent takes to triage one email."""
    category: str = Field(
        ...,
        description="Email category: 'spam', 'urgent', 'normal', 'newsletter', 'internal'"
    )
    priority: int = Field(
        ...,
        ge=1, le=5,
        description="Priority 1 (highest) to 5 (lowest)"
    )
    department: str = Field(
        ...,
        description="Route to: 'sales', 'support', 'engineering', 'hr', 'management', 'ignore'"
    )
    should_reply: bool = Field(
        ...,
        description="Whether this email requires a reply"
    )


# ── Reward / Step response ────────────────────────────────────────────────────
class StepResult(BaseModel):
    observation: Optional[EmailObservation]
    reward: float
    done: bool
    info: Dict[str, Any]


# ── State ─────────────────────────────────────────────────────────────────────
class EnvState(BaseModel):
    episode_id: str
    task_id: str
    current_step: int
    max_steps: int
    total_reward: float
    done: bool
    emails_triaged: int
    correct_decisions: int
    inbox: List[Dict[str, Any]]


# ── Task info ─────────────────────────────────────────────────────────────────
class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    action_schema: Dict[str, Any]
    success_threshold: float


# ── Grader request ────────────────────────────────────────────────────────────
class GraderRequest(BaseModel):
    episode_id: str
    task_id: str


class GraderResult(BaseModel):
    episode_id: str
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    summary: str
