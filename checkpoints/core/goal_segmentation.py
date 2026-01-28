"""Goal Segmentation implementation for MindTheGoal framework."""

from typing import List, Optional
from core.models import Turn, Goal, Session, TurnQuality


class GoalSegmenter:
    """
    Segments conversation turns into distinct goals.
    
    A goal represents a coherent user intent or information need.
    Goal boundaries are detected when users start new topics/tasks.
    """

    def segment_session(self, session: Session) -> Session:
        """
        Segment a session's turns into goals.
        
        Uses the is_new_goal flag on turns (set by LLM judge)
        to determine goal boundaries.
        
        Args:
            session: Session with evaluated turns
            
        Returns:
            Session with populated goals list
        """
        if not session.turns:
            return session
        
        goals: List[Goal] = []
        current_goal_turns: List[Turn] = []
        goal_number = 1
        
        for i, turn in enumerate(session.turns):
            # First turn always starts a new goal
            if i == 0:
                turn.is_new_goal = True
            
            # If this turn starts a new goal and we have accumulated turns
            if turn.is_new_goal and current_goal_turns:
                # Create goal from accumulated turns
                goal = self._create_goal(current_goal_turns, goal_number)
                goals.append(goal)
                goal_number += 1
                current_goal_turns = []
            
            current_goal_turns.append(turn)
        
        # Don't forget the last goal
        if current_goal_turns:
            goal = self._create_goal(current_goal_turns, goal_number)
            goals.append(goal)
        
        session.goals = goals
        return session

    def segment_from_evaluations(
        self,
        turns: List[Turn],
        new_goal_flags: List[bool]
    ) -> List[Goal]:
        """
        Segment turns using explicit new_goal flags.
        
        Args:
            turns: List of turns
            new_goal_flags: Boolean flags indicating new goal starts
            
        Returns:
            List of Goal objects
        """
        if len(turns) != len(new_goal_flags):
            raise ValueError("turns and new_goal_flags must have same length")
        
        # Apply flags to turns
        for turn, is_new_goal in zip(turns, new_goal_flags):
            turn.is_new_goal = is_new_goal
        
        # First turn is always a new goal
        if turns:
            turns[0].is_new_goal = True
        
        goals: List[Goal] = []
        current_goal_turns: List[Turn] = []
        goal_number = 1
        
        for turn in turns:
            if turn.is_new_goal and current_goal_turns:
                goal = self._create_goal(current_goal_turns, goal_number)
                goals.append(goal)
                goal_number += 1
                current_goal_turns = []
            
            current_goal_turns.append(turn)
        
        if current_goal_turns:
            goal = self._create_goal(current_goal_turns, goal_number)
            goals.append(goal)
        
        return goals

    def _create_goal(self, turns: List[Turn], goal_number: int) -> Goal:
        """Create a Goal from a list of turns."""
        goal = Goal(
            goal_number=goal_number,
            turns=turns
        )
        goal.compute_quality()
        return goal

    def heuristic_segment(
        self,
        turns: List[Turn],
        similarity_threshold: float = 0.3
    ) -> List[Goal]:
        """
        Heuristic-based segmentation without LLM.
        
        Uses simple rules:
        - Greeting patterns start new goals
        - Topic change keywords start new goals
        - Very different message length might indicate new goal
        
        Args:
            turns: List of turns to segment
            similarity_threshold: Threshold for similarity (not used yet)
            
        Returns:
            List of Goals
        """
        NEW_GOAL_PATTERNS = [
            "i have another", "new question", "different topic",
            "also", "by the way", "one more thing", "additionally",
            "separate question", "unrelated", "moving on",
            "can you also", "i also need", "another thing"
        ]
        
        GREETING_PATTERNS = [
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "hi there", "hello there"
        ]
        
        goals: List[Goal] = []
        current_goal_turns: List[Turn] = []
        goal_number = 1
        
        for i, turn in enumerate(turns):
            is_new_goal = False
            user_msg_lower = turn.user_message.lower().strip()
            
            # First turn is always new goal
            if i == 0:
                is_new_goal = True
            else:
                # Check for new goal patterns
                for pattern in NEW_GOAL_PATTERNS:
                    if pattern in user_msg_lower:
                        is_new_goal = True
                        break
                
                # Check for greeting (might indicate conversation restart)
                if not is_new_goal:
                    for greeting in GREETING_PATTERNS:
                        if user_msg_lower.startswith(greeting):
                            is_new_goal = True
                            break
            
            turn.is_new_goal = is_new_goal
            
            if is_new_goal and current_goal_turns:
                goal = self._create_goal(current_goal_turns, goal_number)
                goals.append(goal)
                goal_number += 1
                current_goal_turns = []
            
            current_goal_turns.append(turn)
        
        if current_goal_turns:
            goal = self._create_goal(current_goal_turns, goal_number)
            goals.append(goal)
        
        return goals
