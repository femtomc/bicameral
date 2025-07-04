"""Complete interaction model for user query → AI interpretation → actions → feedback cycles."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FeedbackType(str, Enum):
    """Types of user feedback after an interaction."""

    APPROVAL = "approval"  # "Great, thanks!"
    CORRECTION = "correction"  # "No, I meant X not Y"
    FOLLOWUP = "followup"  # "Also do Z"
    CLARIFICATION = "clarification"  # "What I meant was..."
    NONE = "none"  # No explicit feedback


class ActionStatus(str, Enum):
    """Status of an individual action."""

    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Action:
    """A single action within an interaction."""

    action_type: str  # "edit_file", "search_code", etc.
    target: Optional[str] = None  # file_path or resource
    details: Dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PLANNED
    result: Optional[str] = None  # What happened
    error: Optional[str] = None  # Error message if failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate action duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "action_type": self.action_type,
            "target": self.target,
            "details": self.details,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
        }


@dataclass
class Interaction:
    """A complete interaction cycle between user and AI."""

    # Required fields (no defaults)
    interaction_id: str
    session_id: str
    user_query: str  # Original natural language query

    # Optional fields (with defaults)
    timestamp: datetime = field(default_factory=datetime.now)
    query_context: Dict[str, Any] = field(default_factory=dict)  # Files open, recent actions, etc.

    # AI Processing
    ai_interpretation: Optional[str] = None  # What AI understood
    planned_actions: List[str] = field(default_factory=list)  # Action types AI plans to take
    confidence: float = 0.0  # AI's confidence in interpretation
    active_role: Optional[str] = None  # Which role was active

    # Execution
    actions_taken: List[Action] = field(default_factory=list)
    execution_started_at: Optional[datetime] = None
    execution_completed_at: Optional[datetime] = None
    tokens_used: int = 0

    # Outcome
    user_feedback: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.NONE
    success: bool = False

    # Metadata
    tags: List[str] = field(default_factory=list)  # For categorization
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total execution time in seconds."""
        if self.execution_started_at and self.execution_completed_at:
            return (self.execution_completed_at - self.execution_started_at).total_seconds()
        return None

    @property
    def was_corrected(self) -> bool:
        """Check if user corrected the AI's interpretation."""
        return self.feedback_type == FeedbackType.CORRECTION

    @property
    def action_sequence(self) -> List[str]:
        """Get the sequence of action types taken."""
        return [action.action_type for action in self.actions_taken]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_query": self.user_query,
            "query_context": self.query_context,
            "ai_interpretation": self.ai_interpretation,
            "planned_actions": self.planned_actions,
            "confidence": self.confidence,
            "active_role": self.active_role,
            "actions_taken": [a.to_dict() for a in self.actions_taken],
            "execution_started_at": (
                self.execution_started_at.isoformat() if self.execution_started_at else None
            ),
            "execution_completed_at": (
                self.execution_completed_at.isoformat() if self.execution_completed_at else None
            ),
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "user_feedback": self.user_feedback,
            "feedback_type": self.feedback_type.value,
            "success": self.success,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Create from dictionary."""
        interaction = cls(
            interaction_id=data["interaction_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_query=data["user_query"],
            query_context=data.get("query_context", {}),
            ai_interpretation=data.get("ai_interpretation"),
            planned_actions=data.get("planned_actions", []),
            confidence=data.get("confidence", 0.0),
            active_role=data.get("active_role"),
            tokens_used=data.get("tokens_used", 0),
            user_feedback=data.get("user_feedback"),
            feedback_type=FeedbackType(data.get("feedback_type", "none")),
            success=data.get("success", False),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

        # Reconstruct actions
        for action_data in data.get("actions_taken", []):
            action = Action(
                action_type=action_data["action_type"],
                target=action_data.get("target"),
                details=action_data.get("details", {}),
                status=ActionStatus(action_data.get("status", "completed")),
                result=action_data.get("result"),
                error=action_data.get("error"),
            )
            if action_data.get("started_at"):
                action.started_at = datetime.fromisoformat(action_data["started_at"])
            if action_data.get("completed_at"):
                action.completed_at = datetime.fromisoformat(action_data["completed_at"])
            interaction.actions_taken.append(action)

        # Set execution times
        if data.get("execution_started_at"):
            interaction.execution_started_at = datetime.fromisoformat(data["execution_started_at"])
        if data.get("execution_completed_at"):
            interaction.execution_completed_at = datetime.fromisoformat(
                data["execution_completed_at"]
            )

        return interaction


@dataclass
class InteractionPattern:
    """A pattern discovered from multiple interactions."""

    pattern_id: str
    pattern_type: str  # "intent", "workflow", "correction", "success"

    # For intent patterns
    query_patterns: List[str] = field(default_factory=list)  # Common query phrasings
    typical_interpretation: Optional[str] = None
    typical_actions: List[str] = field(default_factory=list)

    # Statistics
    frequency: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    last_seen: Optional[datetime] = None

    # Learning
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
