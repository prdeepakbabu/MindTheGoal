"""Core framework implementation for MindTheGoal evaluation."""

from core.models import (
    TurnQuality,
    RCOF,
    Turn,
    Goal,
    Session,
    TurnEvaluation,
    GoalEvaluation,
    SessionEvaluation,
    EvaluationResult,
    GSRReport,
)
from core.gsr_calculator import GSRCalculator
from core.rcof_classifier import RCOFClassifier
from core.goal_segmentation import GoalSegmenter

__all__ = [
    # Enums
    "TurnQuality",
    "RCOF",
    # Data models
    "Turn",
    "Goal", 
    "Session",
    "TurnEvaluation",
    "GoalEvaluation",
    "SessionEvaluation",
    "EvaluationResult",
    "GSRReport",
    # Core components
    "GSRCalculator",
    "RCOFClassifier",
    "GoalSegmenter",
]
