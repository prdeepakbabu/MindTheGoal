"""Tests for core framework components."""

import pytest
from datetime import datetime

from core.models import (
    Turn, Goal, Session, TurnQuality, RCOF,
    TurnEvaluation, EvaluationResult, GSRReport
)
from core.gsr_calculator import GSRCalculator
from core.rcof_classifier import RCOFClassifier
from core.goal_segmentation import GoalSegmenter


class TestModels:
    """Tests for core data models."""
    
    def test_turn_creation(self):
        """Test Turn creation with defaults."""
        turn = Turn(
            turn_number=1,
            user_message="Hello",
            agent_response="Hi there!"
        )
        
        assert turn.turn_number == 1
        assert turn.user_message == "Hello"
        assert turn.agent_response == "Hi there!"
        assert turn.quality == TurnQuality.PENDING
        assert turn.rcof is None
        assert turn.is_new_goal is False
    
    def test_turn_to_dict(self):
        """Test Turn serialization."""
        turn = Turn(
            turn_number=1,
            user_message="Test",
            agent_response="Response",
            quality=TurnQuality.SUCCESS
        )
        
        data = turn.to_dict()
        
        assert data["turn_number"] == 1
        assert data["quality"] == "success"
    
    def test_turn_from_dict(self):
        """Test Turn deserialization."""
        data = {
            "turn_number": 2,
            "user_message": "Question",
            "agent_response": "Answer",
            "quality": "failure",
            "rcof": "E1"
        }
        
        turn = Turn.from_dict(data)
        
        assert turn.turn_number == 2
        assert turn.quality == TurnQuality.FAILURE
        assert turn.rcof == "E1"
    
    def test_goal_success_all_turns_successful(self):
        """Test goal success when all turns are successful."""
        turns = [
            Turn(turn_number=1, user_message="Q1", agent_response="A1", quality=TurnQuality.SUCCESS),
            Turn(turn_number=2, user_message="Q2", agent_response="A2", quality=TurnQuality.SUCCESS),
        ]
        
        goal = Goal(goal_number=1, turns=turns)
        
        assert goal.is_successful is True
    
    def test_goal_failure_if_any_turn_fails(self):
        """Test goal failure if any turn fails (strict criteria)."""
        turns = [
            Turn(turn_number=1, user_message="Q1", agent_response="A1", quality=TurnQuality.SUCCESS),
            Turn(turn_number=2, user_message="Q2", agent_response="A2", quality=TurnQuality.FAILURE, rcof="E1"),
        ]
        
        goal = Goal(goal_number=1, turns=turns)
        
        assert goal.is_successful is False
    
    def test_goal_compute_quality(self):
        """Test automatic quality computation."""
        turns = [
            Turn(turn_number=1, user_message="Q", agent_response="A", quality=TurnQuality.FAILURE, rcof="E3"),
        ]
        
        goal = Goal(goal_number=1, turns=turns)
        goal.compute_quality()
        
        assert goal.quality == TurnQuality.FAILURE
        assert goal.rcof == "E3"
    
    def test_session_gsr_calculation(self):
        """Test session GSR calculation."""
        # Goals need turns to be counted as successful
        goal1 = Goal(goal_number=1, turns=[Turn(turn_number=1, quality=TurnQuality.SUCCESS)])
        goal2 = Goal(goal_number=2, turns=[Turn(turn_number=1, quality=TurnQuality.FAILURE)])
        goal3 = Goal(goal_number=3, turns=[Turn(turn_number=1, quality=TurnQuality.SUCCESS)])
        goal4 = Goal(goal_number=4, turns=[Turn(turn_number=1, quality=TurnQuality.SUCCESS)])
        
        session = Session(goals=[goal1, goal2, goal3, goal4])
        
        assert session.gsr == 75.0  # 3/4 = 75%
    
    def test_rcof_descriptions(self):
        """Test RCOF code descriptions."""
        desc = RCOF.get_description("E1")
        assert "Language Understanding" in desc
        
        desc = RCOF.get_description("E5")
        assert "System Error" in desc


class TestGSRCalculator:
    """Tests for GSR calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = GSRCalculator()
    
    def test_calculate_goal_quality_success(self):
        """Test successful goal quality calculation."""
        turns = [
            Turn(turn_number=1, quality=TurnQuality.SUCCESS),
            Turn(turn_number=2, quality=TurnQuality.SUCCESS),
        ]
        goal = Goal(goal_number=1, turns=turns)
        
        quality, rcof = self.calculator.calculate_goal_quality(goal)
        
        assert quality == TurnQuality.SUCCESS
        assert rcof is None
    
    def test_calculate_goal_quality_failure(self):
        """Test failed goal quality calculation."""
        turns = [
            Turn(turn_number=1, quality=TurnQuality.SUCCESS),
            Turn(turn_number=2, quality=TurnQuality.FAILURE, rcof="E2"),
        ]
        goal = Goal(goal_number=1, turns=turns)
        
        quality, rcof = self.calculator.calculate_goal_quality(goal)
        
        assert quality == TurnQuality.FAILURE
        assert rcof == "E2"
    
    def test_calculate_session_gsr(self):
        """Test session GSR calculation."""
        session = Session(goals=[
            Goal(goal_number=1, quality=TurnQuality.SUCCESS),
            Goal(goal_number=2, quality=TurnQuality.SUCCESS),
            Goal(goal_number=3, quality=TurnQuality.FAILURE),
        ])
        
        gsr = self.calculator.calculate_session_gsr(session)
        
        assert gsr == pytest.approx(66.67, rel=0.01)
    
    def test_calculate_dataset_gsr(self):
        """Test dataset-level GSR calculation."""
        sessions = [
            Session(goals=[
                Goal(goal_number=1, quality=TurnQuality.SUCCESS, turns=[Turn(turn_number=1)]),
                Goal(goal_number=2, quality=TurnQuality.SUCCESS, turns=[Turn(turn_number=1)]),
            ]),
            Session(goals=[
                Goal(goal_number=1, quality=TurnQuality.FAILURE, rcof="E1", turns=[Turn(turn_number=1)]),
            ]),
        ]
        
        result = self.calculator.calculate_dataset_gsr(sessions)
        
        assert result.total_sessions == 2
        assert result.total_goals == 3
        assert result.overall_gsr == pytest.approx(66.67, rel=0.01)
        assert result.rcof_distribution.get("E1", 0) == 1
    
    def test_generate_report(self):
        """Test report generation."""
        result = EvaluationResult(
            dataset_name="test",
            total_sessions=10,
            total_goals=50,
            total_turns=100,
            overall_gsr=80.0,
            single_turn_gsr=85.0,
            multi_turn_gsr=75.0,
            rcof_distribution={"E1": 5, "E3": 3},
            sessions=[]
        )
        
        report = self.calculator.generate_report(result)
        
        assert report.dataset_name == "test"
        assert report.overall_gsr == 80.0


class TestRCOFClassifier:
    """Tests for RCOF classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = RCOFClassifier()
    
    def test_classify_explicit_code(self):
        """Test classification with explicit code."""
        turn = Turn(turn_number=1, quality=TurnQuality.FAILURE)
        
        code = self.classifier.classify(turn, explicit_code="E4")
        
        assert code == "E4"
    
    def test_classify_from_reasoning_refusal(self):
        """Test classification from judge reasoning - refusal pattern."""
        turn = Turn(
            turn_number=1, 
            quality=TurnQuality.FAILURE,
            agent_response="I'm sorry, I cannot help with that request."
        )
        
        code = self.classifier.classify(turn)
        
        # Agent response with "cannot help" pattern should classify as E2
        assert code == "E2"
    
    def test_classify_from_response_retrieval(self):
        """Test classification from response content - retrieval failure."""
        turn = Turn(
            turn_number=1,
            quality=TurnQuality.FAILURE,
            agent_response="I couldn't find any information about that."
        )
        
        code = self.classifier.classify(turn)
        
        assert code == "E4"
    
    def test_classify_default_e1(self):
        """Test default classification to E1."""
        turn = Turn(
            turn_number=1,
            quality=TurnQuality.FAILURE,
            agent_response="Here is some response"
        )
        
        code = self.classifier.classify(turn)
        
        assert code == "E1"
    
    def test_classify_successful_turn(self):
        """Test no classification for successful turn."""
        turn = Turn(turn_number=1, quality=TurnQuality.SUCCESS)
        
        code = self.classifier.classify(turn)
        
        assert code is None
    
    def test_valid_code_check(self):
        """Test valid RCOF code validation."""
        assert self.classifier.is_valid_code("E1") is True
        assert self.classifier.is_valid_code("E7") is True
        assert self.classifier.is_valid_code("E8") is False
        assert self.classifier.is_valid_code("invalid") is False


class TestGoalSegmenter:
    """Tests for goal segmentation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.segmenter = GoalSegmenter()
    
    def test_segment_single_goal(self):
        """Test segmentation with single goal."""
        turns = [
            Turn(turn_number=1, user_message="Question 1", is_new_goal=True),
            Turn(turn_number=2, user_message="Follow up"),
        ]
        session = Session(turns=turns)
        
        session = self.segmenter.segment_session(session)
        
        assert len(session.goals) == 1
        assert session.goals[0].turn_count == 2
    
    def test_segment_multiple_goals(self):
        """Test segmentation with multiple goals."""
        turns = [
            Turn(turn_number=1, user_message="First question", is_new_goal=True),
            Turn(turn_number=2, user_message="Follow up"),
            Turn(turn_number=3, user_message="New topic entirely", is_new_goal=True),
            Turn(turn_number=4, user_message="More about new topic"),
        ]
        session = Session(turns=turns)
        
        session = self.segmenter.segment_session(session)
        
        assert len(session.goals) == 2
        assert session.goals[0].turn_count == 2
        assert session.goals[1].turn_count == 2
    
    def test_segment_from_evaluations(self):
        """Test segmentation from explicit flags."""
        turns = [
            Turn(turn_number=1, user_message="Q1"),
            Turn(turn_number=2, user_message="Q2"),
            Turn(turn_number=3, user_message="Q3"),
        ]
        flags = [True, False, True]
        
        goals = self.segmenter.segment_from_evaluations(turns, flags)
        
        assert len(goals) == 2
        assert goals[0].turn_count == 2
        assert goals[1].turn_count == 1
    
    def test_heuristic_segment_greeting(self):
        """Test heuristic segmentation detects greetings."""
        turns = [
            Turn(turn_number=1, user_message="I need a restaurant"),
            Turn(turn_number=2, user_message="Something cheap please"),
            Turn(turn_number=3, user_message="Hello, I need a hotel now"),
        ]
        
        goals = self.segmenter.heuristic_segment(turns)
        
        assert len(goals) == 2
    
    def test_heuristic_segment_new_topic(self):
        """Test heuristic segmentation detects new topic patterns."""
        turns = [
            Turn(turn_number=1, user_message="Find me a taxi"),
            Turn(turn_number=2, user_message="I have another question about trains"),
        ]
        
        goals = self.segmenter.heuristic_segment(turns)
        
        assert len(goals) == 2


class TestTurnEvaluation:
    """Tests for turn evaluation data class."""
    
    def test_evaluation_to_dict(self):
        """Test evaluation serialization."""
        evaluation = TurnEvaluation(
            turn_number=1,
            is_new_goal=True,
            quality=TurnQuality.SUCCESS,
            reasoning="Good response"
        )
        
        data = evaluation.to_dict()
        
        assert data["turn_number"] == 1
        assert data["is_new_goal"] is True
        assert data["quality"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
