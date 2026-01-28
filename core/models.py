"""Core data models for the MindTheGoal evaluation framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class TurnQuality(str, Enum):
    """Quality classification for a turn."""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


class RCOF(str, Enum):
    """Root Cause of Failure taxonomy from the paper (E1-E7)."""
    E1_LANGUAGE_UNDERSTANDING = "E1"
    E2_REFUSAL_TO_ANSWER = "E2"
    E3_INCORRECT_RETRIEVAL = "E3"
    E4_RETRIEVAL_FAILURE = "E4"
    E5_SYSTEM_ERROR = "E5"
    E6_INCORRECT_ROUTING = "E6"
    E7_OUT_OF_DOMAIN = "E7"

    @classmethod
    def get_description(cls, code: str) -> str:
        """Get human-readable description for an RCOF code."""
        descriptions = {
            "E1": "Language Understanding Failure - Misunderstood user's request or context",
            "E2": "Refusal to Answer - Inappropriate refusal despite ability to help",
            "E3": "Incorrect Retrieval - Retrieved wrong informational content",
            "E4": "Retrieval Failure - Failed to retrieve any relevant information",
            "E5": "System Error - Technical issues (timeout, truncation, integration failure)",
            "E6": "Incorrect Routing - Query routed to wrong domain/module",
            "E7": "Out-of-Domain - Request outside system's designed scope"
        }
        return descriptions.get(code, "Unknown error code")

    @classmethod
    def get_all_descriptions(cls) -> Dict[str, str]:
        """Get all RCOF codes with descriptions."""
        return {member.value: cls.get_description(member.value) for member in cls}


@dataclass
class Turn:
    """A single turn in a conversation (user query + agent response)."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int = 0
    user_message: str = ""
    agent_response: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Evaluation fields
    quality: TurnQuality = TurnQuality.PENDING
    rcof: Optional[str] = None
    is_new_goal: bool = False
    
    # Judge output
    judge_reasoning: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "agent_response": self.agent_response,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "quality": self.quality.value if self.quality else None,
            "rcof": self.rcof,
            "is_new_goal": self.is_new_goal,
            "judge_reasoning": self.judge_reasoning,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        """Create from dictionary."""
        return cls(
            turn_id=data.get("turn_id", str(uuid.uuid4())),
            turn_number=data.get("turn_number", 0),
            user_message=data.get("user_message", ""),
            agent_response=data.get("agent_response", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            quality=TurnQuality(data["quality"]) if data.get("quality") else TurnQuality.PENDING,
            rcof=data.get("rcof"),
            is_new_goal=data.get("is_new_goal", False),
            judge_reasoning=data.get("judge_reasoning"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Goal:
    """A user goal comprising one or more turns."""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_number: int = 0
    turns: List[Turn] = field(default_factory=list)
    
    # Computed/stored fields
    quality: TurnQuality = TurnQuality.PENDING
    rcof: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if all turns in goal are successful (strict criteria from paper)."""
        if not self.turns:
            return False
        return all(t.quality == TurnQuality.SUCCESS for t in self.turns)

    @property
    def turn_count(self) -> int:
        """Get number of turns in this goal."""
        return len(self.turns)

    @property
    def is_multi_turn(self) -> bool:
        """Check if this is a multi-turn goal."""
        return len(self.turns) > 1

    @property
    def first_failed_turn(self) -> Optional[Turn]:
        """Get the first failed turn (for RCOF attribution)."""
        for turn in self.turns:
            if turn.quality == TurnQuality.FAILURE:
                return turn
        return None

    def compute_quality(self) -> None:
        """Compute goal quality based on turns."""
        if not self.turns:
            self.quality = TurnQuality.PENDING
            self.rcof = None
            return
        
        if self.is_successful:
            self.quality = TurnQuality.SUCCESS
            self.rcof = None
        else:
            self.quality = TurnQuality.FAILURE
            failed_turn = self.first_failed_turn
            if failed_turn:
                self.rcof = failed_turn.rcof

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal_id": self.goal_id,
            "goal_number": self.goal_number,
            "turns": [t.to_dict() for t in self.turns],
            "quality": self.quality.value if self.quality else None,
            "rcof": self.rcof,
            "is_successful": self.is_successful,
            "turn_count": self.turn_count,
            "is_multi_turn": self.is_multi_turn,
            "metadata": self.metadata
        }


@dataclass
class Session:
    """A complete conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goals: List[Goal] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)  # Raw turns before segmentation
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        """Get total number of turns across all goals."""
        if self.goals:
            return sum(g.turn_count for g in self.goals)
        return len(self.turns)

    @property
    def total_goals(self) -> int:
        """Get number of goals."""
        return len(self.goals)

    @property
    def successful_goals(self) -> int:
        """Get number of successful goals."""
        return sum(1 for g in self.goals if g.is_successful)

    @property
    def failed_goals(self) -> int:
        """Get number of failed goals."""
        return sum(1 for g in self.goals if not g.is_successful and g.quality != TurnQuality.PENDING)

    @property
    def gsr(self) -> float:
        """Calculate Goal Success Rate for this session."""
        if not self.goals:
            return 0.0
        return (self.successful_goals / len(self.goals)) * 100

    @property
    def single_turn_goals(self) -> List[Goal]:
        """Get single-turn goals."""
        return [g for g in self.goals if not g.is_multi_turn]

    @property
    def multi_turn_goals(self) -> List[Goal]:
        """Get multi-turn goals."""
        return [g for g in self.goals if g.is_multi_turn]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "goals": [g.to_dict() for g in self.goals],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_turns": self.total_turns,
            "total_goals": self.total_goals,
            "successful_goals": self.successful_goals,
            "gsr": self.gsr,
            "metadata": self.metadata
        }


@dataclass
class TurnEvaluation:
    """Evaluation result for a single turn from the judge."""
    turn_number: int
    is_new_goal: bool
    quality: TurnQuality
    rcof: Optional[str] = None
    reasoning: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_number": self.turn_number,
            "is_new_goal": self.is_new_goal,
            "quality": self.quality.value,
            "rcof": self.rcof,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


@dataclass
class GoalEvaluation:
    """Evaluation result for a goal."""
    goal_number: int
    turns: List[TurnEvaluation]
    quality: TurnQuality
    rcof: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.quality == TurnQuality.SUCCESS


@dataclass
class SessionEvaluation:
    """Complete evaluation for a session."""
    session_id: str
    goals: List[GoalEvaluation]
    gsr: float
    total_turns: int
    total_goals: int
    successful_goals: int
    failed_goals: int


@dataclass
class EvaluationResult:
    """Complete evaluation result for a dataset."""
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = ""
    total_sessions: int = 0
    total_goals: int = 0
    total_turns: int = 0
    
    overall_gsr: float = 0.0
    turn_success_rate: float = 0.0  # successful turns / total turns
    single_turn_gsr: float = 0.0    # GSR for sessions with 1 turn
    multi_turn_gsr: float = 0.0     # GSR for sessions with >1 turn
    
    successful_turns: int = 0
    single_turn_session_count: int = 0
    multi_turn_session_count: int = 0
    
    rcof_distribution: Dict[str, int] = field(default_factory=dict)
    domain_gsr: Dict[str, float] = field(default_factory=dict)
    
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    sessions: List[Session] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "dataset_name": self.dataset_name,
            "total_sessions": self.total_sessions,
            "total_goals": self.total_goals,
            "total_turns": self.total_turns,
            "overall_gsr": self.overall_gsr,
            "turn_success_rate": self.turn_success_rate,
            "single_turn_gsr": self.single_turn_gsr,
            "multi_turn_gsr": self.multi_turn_gsr,
            "successful_turns": self.successful_turns,
            "single_turn_session_count": self.single_turn_session_count,
            "multi_turn_session_count": self.multi_turn_session_count,
            "rcof_distribution": self.rcof_distribution,
            "domain_gsr": self.domain_gsr,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None
        }


@dataclass
class GSRReport:
    """GSR evaluation report for API responses."""
    evaluation_id: str
    dataset_name: str
    overall_gsr: float
    turn_success_rate: float  # successful turns / total turns
    single_turn_gsr: float    # GSR for sessions with 1 turn
    multi_turn_gsr: float     # GSR for sessions with >1 turn
    total_goals: int
    total_sessions: int
    total_turns: int
    successful_turns: int = 0
    single_turn_session_count: int = 0
    multi_turn_session_count: int = 0
    successful_goals: int = 0
    failed_goals: int = 0
    rcof_distribution: Dict[str, int] = field(default_factory=dict)
    rcof_percentages: Dict[str, float] = field(default_factory=dict)
    evaluated_at: str = ""
    
    def __post_init__(self):
        """Calculate RCOF percentages after initialization."""
        total_failures = sum(self.rcof_distribution.values())
        if total_failures > 0:
            self.rcof_percentages = {
                code: (count / total_failures) * 100
                for code, count in self.rcof_distribution.items()
            }
