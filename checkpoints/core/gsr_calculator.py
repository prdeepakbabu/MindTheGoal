"""GSR (Goal Success Rate) Calculator implementation."""

from typing import List, Dict, Tuple
from collections import Counter

from core.models import (
    Session,
    Goal,
    Turn,
    TurnQuality,
    EvaluationResult,
    GSRReport,
)


class GSRCalculator:
    """
    Calculator for Goal Success Rate (GSR) metrics.
    
    Implements the strict success criteria from the paper:
    A goal is successful IFF all its turns are successful.
    """

    def calculate_goal_quality(self, goal: Goal) -> Tuple[TurnQuality, str]:
        """
        Calculate quality for a single goal.
        
        Returns:
            Tuple of (quality, rcof_code or None)
        """
        if not goal.turns:
            return TurnQuality.PENDING, None
        
        # Check if all turns are successful
        all_successful = all(
            turn.quality == TurnQuality.SUCCESS 
            for turn in goal.turns
        )
        
        if all_successful:
            return TurnQuality.SUCCESS, None
        
        # Find first failed turn for RCOF attribution
        for turn in goal.turns:
            if turn.quality == TurnQuality.FAILURE:
                return TurnQuality.FAILURE, turn.rcof
        
        # If some turns are pending
        return TurnQuality.PENDING, None

    def calculate_session_gsr(self, session: Session) -> float:
        """
        Calculate GSR for a single session.
        
        Returns:
            GSR percentage (0-100)
        """
        if not session.goals:
            return 0.0
        
        successful_goals = sum(
            1 for goal in session.goals 
            if goal.quality == TurnQuality.SUCCESS
        )
        
        return (successful_goals / len(session.goals)) * 100

    def calculate_dataset_gsr(self, sessions: List[Session]) -> EvaluationResult:
        """
        Calculate comprehensive GSR metrics for a dataset.
        
        Args:
            sessions: List of evaluated sessions
            
        Returns:
            EvaluationResult with all metrics
        """
        all_goals: List[Goal] = []
        rcof_counts: Dict[str, int] = Counter()
        domain_goals: Dict[str, List[Goal]] = {}
        
        total_turns = 0
        successful_turns = 0
        
        # Separate sessions by turn count
        single_turn_sessions: List[Session] = []  # Sessions with exactly 1 turn
        multi_turn_sessions: List[Session] = []   # Sessions with >1 turn
        
        for session in sessions:
            # Count total turns in session
            session_turn_count = len(session.turns)
            
            # Categorize session by turn count
            if session_turn_count == 1:
                single_turn_sessions.append(session)
            else:
                multi_turn_sessions.append(session)
            
            # Count successful/total turns for Turn Success Rate
            for turn in session.turns:
                total_turns += 1
                if turn.quality == TurnQuality.SUCCESS:
                    successful_turns += 1
            
            # Process goals
            for goal in session.goals:
                all_goals.append(goal)
                
                # Count RCOF for failures
                if goal.quality == TurnQuality.FAILURE and goal.rcof:
                    rcof_counts[goal.rcof] += 1
                
                # Group by domain if available
                domain = session.metadata.get("domain") or goal.metadata.get("domain")
                if domain:
                    if domain not in domain_goals:
                        domain_goals[domain] = []
                    domain_goals[domain].append(goal)
        
        # Calculate overall GSR (goal-level)
        overall_gsr = self._calculate_gsr(all_goals)
        
        # Calculate Turn Success Rate (turn-level)
        turn_success_rate = (successful_turns / total_turns * 100) if total_turns > 0 else 0.0
        
        # Calculate Single-turn Session GSR (sessions with exactly 1 turn)
        single_turn_gsr = self._calculate_session_gsr_list(single_turn_sessions)
        
        # Calculate Multi-turn Session GSR (sessions with >1 turn)
        multi_turn_gsr = self._calculate_session_gsr_list(multi_turn_sessions)
        
        # Calculate per-domain GSR
        domain_gsr = {
            domain: self._calculate_gsr(goals)
            for domain, goals in domain_goals.items()
        }
        
        return EvaluationResult(
            dataset_name="",  # Set by caller
            total_sessions=len(sessions),
            total_goals=len(all_goals),
            total_turns=total_turns,
            overall_gsr=overall_gsr,
            turn_success_rate=turn_success_rate,
            single_turn_gsr=single_turn_gsr,
            multi_turn_gsr=multi_turn_gsr,
            successful_turns=successful_turns,
            single_turn_session_count=len(single_turn_sessions),
            multi_turn_session_count=len(multi_turn_sessions),
            rcof_distribution=dict(rcof_counts),
            domain_gsr=domain_gsr,
            sessions=sessions
        )

    def _calculate_gsr(self, goals: List[Goal]) -> float:
        """Calculate GSR for a list of goals."""
        if not goals:
            return 0.0
        
        successful = sum(
            1 for goal in goals 
            if goal.quality == TurnQuality.SUCCESS
        )
        
        return (successful / len(goals)) * 100

    def _calculate_session_gsr_list(self, sessions: List[Session]) -> float:
        """Calculate average GSR across a list of sessions."""
        if not sessions:
            return 0.0
        
        # Collect all goals from these sessions
        all_goals = []
        for session in sessions:
            all_goals.extend(session.goals)
        
        return self._calculate_gsr(all_goals)

    def generate_report(self, result: EvaluationResult) -> GSRReport:
        """
        Generate a GSR report from evaluation results.
        
        Args:
            result: EvaluationResult from dataset evaluation
            
        Returns:
            GSRReport for API response
        """
        successful_goals = sum(
            1 for session in result.sessions
            for goal in session.goals
            if goal.quality == TurnQuality.SUCCESS
        )
        
        failed_goals = sum(
            1 for session in result.sessions
            for goal in session.goals
            if goal.quality == TurnQuality.FAILURE
        )
        
        return GSRReport(
            evaluation_id=result.evaluation_id,
            dataset_name=result.dataset_name,
            overall_gsr=round(result.overall_gsr, 2),
            turn_success_rate=round(result.turn_success_rate, 2),
            single_turn_gsr=round(result.single_turn_gsr, 2),
            multi_turn_gsr=round(result.multi_turn_gsr, 2),
            total_goals=result.total_goals,
            total_sessions=result.total_sessions,
            total_turns=result.total_turns,
            successful_turns=result.successful_turns,
            single_turn_session_count=result.single_turn_session_count,
            multi_turn_session_count=result.multi_turn_session_count,
            successful_goals=successful_goals,
            failed_goals=failed_goals,
            rcof_distribution=result.rcof_distribution,
            evaluated_at=result.evaluated_at.isoformat() if result.evaluated_at else ""
        )

    def print_report(self, report: GSRReport) -> str:
        """
        Generate a formatted text report.
        
        Args:
            report: GSRReport to format
            
        Returns:
            Formatted string report
        """
        lines = [
            "═" * 65,
            "                MindTheGoal Evaluation Report",
            "═" * 65,
            "",
            f"Dataset: {report.dataset_name}",
            f"Evaluation ID: {report.evaluation_id}",
            f"Evaluated at: {report.evaluated_at}",
            "",
            "─" * 65,
            "                      Summary Metrics",
            "─" * 65,
            "",
            f"Total Sessions:       {report.total_sessions:,}",
            f"  - Single-turn:      {report.single_turn_session_count:,}",
            f"  - Multi-turn:       {report.multi_turn_session_count:,}",
            "",
            f"Total Goals:          {report.total_goals:,}",
            f"Total Turns:          {report.total_turns:,}",
            "",
            f"Successful Goals:     {report.successful_goals:,}",
            f"Failed Goals:         {report.failed_goals:,}",
            f"Successful Turns:     {report.successful_turns:,}",
            "",
            "─" * 65,
            "                      Success Metrics",
            "─" * 65,
            "",
            f"Overall GSR:          {report.overall_gsr:.1f}%   (successful goals / total goals)",
            f"Turn Success Rate:    {report.turn_success_rate:.1f}%   (successful turns / total turns)",
            f"Single-turn GSR:      {report.single_turn_gsr:.1f}%   (GSR for 1-turn sessions)",
            f"Multi-turn GSR:       {report.multi_turn_gsr:.1f}%   (GSR for multi-turn sessions)",
            "",
            "─" * 65,
            "                Root Cause of Failure (RCOF)",
            "─" * 65,
            "",
        ]
        
        # Add RCOF distribution
        from core.models import RCOF
        
        for code in ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]:
            count = report.rcof_distribution.get(code, 0)
            percentage = report.rcof_percentages.get(code, 0)
            bar_length = int(percentage / 5)  # Scale to max 20 chars
            bar = "█" * bar_length
            desc = RCOF.get_description(code).split(" - ")[0]
            lines.append(f"{code} - {desc:25} {count:4}  {percentage:5.1f}%  {bar}")
        
        lines.extend([
            "",
            "═" * 65
        ])
        
        return "\n".join(lines)
