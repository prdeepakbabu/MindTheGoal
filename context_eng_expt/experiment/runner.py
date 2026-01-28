"""Experiment runner that orchestrates user-chatbot conversations."""

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..agents.simulated_user import SimulatedUser
from ..agents.chatbot_agent import ChatbotAgent
from ..agents.goal_detector import GoalDetector
from ..tools.router import ToolRouter
from ..context.strategies import (
    ContextStrategy, 
    FullContextStrategy,
    SlidingWindowStrategy,
    GoalBoundaryStrategy,
    TokenBudgetStrategy
)
from ..context.goal_summarizer import GoalSummarizer
from ..config import ExperimentConfig, CompactionStrategy, SCENARIOS, PERSONAS
from .metrics import MetricsCollector, ExperimentMetrics

logger = logging.getLogger(__name__)


@dataclass
class ConversationResult:
    """Result of a single conversation."""
    scenario_id: str
    persona_id: str
    strategy: str
    run_id: int
    
    # Conversation data
    turns: List[Dict[str, str]]
    goals_detected: int
    
    # Metrics
    total_turns: int
    full_context_tokens: int
    compressed_context_tokens: int
    token_reduction_pct: float
    
    # Tool metrics
    tool_calls_total: int = 0
    tool_calls_successful: int = 0
    tool_accuracy: float = 0.0
    tools_used: Dict[str, int] = field(default_factory=dict)
    
    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0


class ConversationOrchestrator:
    """
    Orchestrates a single conversation between simulated user and chatbot.
    """
    
    def __init__(
        self,
        user: SimulatedUser,
        chatbot: ChatbotAgent,
        goal_detector: GoalDetector,
        max_turns: int = 30
    ):
        self.user = user
        self.chatbot = chatbot
        self.goal_detector = goal_detector
        self.max_turns = max_turns
        self.conversation: List[Dict[str, str]] = []
    
    async def run(self) -> Dict[str, Any]:
        """
        Run a complete conversation until goals are achieved or max turns reached.
        """
        turn_count = 0
        agent_response = None
        
        logger.info(f"Starting conversation with {len(self.user.goals)} goals")
        
        while turn_count < self.max_turns:
            # 1. User generates message
            user_message = await self.user.generate_utterance(agent_response)
            logger.info(f"Turn {turn_count+1} User: {user_message[:80]}...")
            
            # 2. Detect if this is a new goal
            is_new_goal = await self.goal_detector.is_new_goal(user_message, agent_response)
            if is_new_goal:
                logger.info(">>> New goal detected!")
            
            # 3. Chatbot responds
            agent_response = await self.chatbot.respond(user_message, is_new_goal=is_new_goal)
            logger.info(f"Turn {turn_count+1} Agent: {agent_response[:80]}...")
            
            # 4. Update goal detector with response
            self.goal_detector.add_agent_response(agent_response)
            
            # 5. Store turn
            self.conversation.append({
                "turn": turn_count + 1,
                "user": user_message,
                "agent": agent_response,
                "is_new_goal": is_new_goal
            })
            
            turn_count += 1
            
            # 6. Check if conversation should end
            if self._should_end(user_message, agent_response):
                logger.info(f"Conversation ended after {turn_count} turns")
                break
            
            # Delay between turns (rate limiting)
            await asyncio.sleep(1.0)
        
        return self._compile_results()
    
    def _should_end(self, user_message: str, agent_response: str) -> bool:
        """Determine if the conversation should end."""
        goodbye_words = ["goodbye", "bye", "that's all", "done", "all set", "nothing else"]
        thank_words = ["thank you", "thanks", "appreciate"]
        user_lower = user_message.lower()
        
        # Strong end signals
        if any(word in user_lower for word in goodbye_words):
            return True
        
        # Thanks + goals achieved
        if any(word in user_lower for word in thank_words):
            if self.user.all_goals_achieved:
                return True
        
        return False
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile conversation results."""
        chatbot_metrics = self.chatbot.get_metrics()
        detector_stats = self.goal_detector.get_stats()
        user_state = self.user.get_state()
        
        return {
            "conversation": self.conversation,
            "total_turns": len(self.conversation),
            "goals_detected": detector_stats["total_goals"],
            "goals_achieved": sum(1 for g in user_state["goal_states"] if g["achieved"]),
            "user_goals": user_state["goals"],
            "chatbot_metrics": chatbot_metrics,
            "detector_stats": detector_stats,
            "context_strategy_stats": (
                self.chatbot.context_strategy.get_stats()
                if self.chatbot.context_strategy else {}
            )
        }


class ExperimentRunner:
    """
    Runs the full experiment matrix across scenarios, personas, and strategies.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ConversationResult] = []
        self.metrics_collector = MetricsCollector()
    
    def _create_strategy(self, strategy_type: CompactionStrategy) -> ContextStrategy:
        """Create a context strategy instance."""
        summarizer = GoalSummarizer()
        
        if strategy_type == CompactionStrategy.FULL_CONTEXT:
            return FullContextStrategy()
        elif strategy_type == CompactionStrategy.SLIDING_WINDOW:
            return SlidingWindowStrategy(window_size=self.config.sliding_window_size)
        elif strategy_type == CompactionStrategy.GOAL_BOUNDARY:
            return GoalBoundaryStrategy(summarizer=summarizer)
        elif strategy_type == CompactionStrategy.TOKEN_BUDGET:
            return TokenBudgetStrategy(
                max_tokens=self.config.token_budget,
                summarizer=summarizer
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
    
    async def run_single(
        self,
        scenario_id: str,
        persona_id: str,
        strategy: CompactionStrategy,
        run_id: int = 1
    ) -> ConversationResult:
        """
        Run a single experiment configuration.
        """
        # Find scenario and persona
        scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
        persona = next((p for p in PERSONAS if p["id"] == persona_id), None)
        
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        if not persona:
            raise ValueError(f"Unknown persona: {persona_id}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {scenario_id} / {persona_id} / {strategy.value} / run {run_id}")
        logger.info(f"Goals: {scenario['goals']}")
        logger.info(f"{'='*60}")
        
        # Create components
        context_strategy = self._create_strategy(strategy)
        tool_router = ToolRouter() if self.config.enable_tools else None
        
        user = SimulatedUser(
            goals=scenario["goals"],
            persona=persona["description"],
            persona_modifier=persona["prompt_modifier"]
        )
        
        chatbot = ChatbotAgent(
            context_strategy=context_strategy,
            tool_router=tool_router
        )
        goal_detector = GoalDetector()
        
        orchestrator = ConversationOrchestrator(
            user=user,
            chatbot=chatbot,
            goal_detector=goal_detector,
            max_turns=self.config.max_turns_per_conversation
        )
        
        # Run conversation
        started_at = datetime.utcnow()
        results = await orchestrator.run()
        completed_at = datetime.utcnow()
        
        # Compile result
        chatbot_metrics = results["chatbot_metrics"]
        tool_metrics = chatbot_metrics.get("tool_metrics", {})
        
        result = ConversationResult(
            scenario_id=scenario_id,
            persona_id=persona_id,
            strategy=strategy.value,
            run_id=run_id,
            turns=results["conversation"],
            goals_detected=results["goals_detected"],
            total_turns=results["total_turns"],
            full_context_tokens=chatbot_metrics["full_context_tokens"],
            compressed_context_tokens=chatbot_metrics["compressed_context_tokens"],
            token_reduction_pct=chatbot_metrics["token_reduction_pct"],
            tool_calls_total=tool_metrics.get("total_calls", 0),
            tool_calls_successful=tool_metrics.get("successful_calls", 0),
            tool_accuracy=tool_metrics.get("accuracy", 0.0),
            tools_used=tool_metrics.get("tools_used", {}),
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds()
        )
        
        self.results.append(result)
        
        # Save if configured
        if self.config.save_conversations:
            self._save_result(result)
        
        logger.info(f"\nResult: {result.total_turns} turns, {result.token_reduction_pct:.1f}% token reduction, {result.tool_accuracy*100:.0f}% tool accuracy")
        
        return result
    
    async def run_matrix(
        self,
        scenarios: Optional[List[str]] = None,
        personas: Optional[List[str]] = None,
        strategies: Optional[List[CompactionStrategy]] = None
    ) -> List[ConversationResult]:
        """
        Run the full experiment matrix.
        """
        scenarios = scenarios or [s["id"] for s in SCENARIOS]
        personas = personas or [p["id"] for p in PERSONAS]
        strategies = strategies or list(CompactionStrategy)
        
        total_runs = (
            len(scenarios) * len(personas) * len(strategies) * 
            self.config.num_runs_per_scenario
        )
        logger.info(f"Starting experiment matrix: {total_runs} total runs")
        
        run_count = 0
        for scenario_id in scenarios:
            for persona_id in personas:
                for strategy in strategies:
                    for run_id in range(1, self.config.num_runs_per_scenario + 1):
                        run_count += 1
                        logger.info(f"\n[{run_count}/{total_runs}]")
                        try:
                            await self.run_single(
                                scenario_id=scenario_id,
                                persona_id=persona_id,
                                strategy=strategy,
                                run_id=run_id
                            )
                        except Exception as e:
                            logger.error(
                                f"Error in {scenario_id}/{persona_id}/{strategy.value}: {e}"
                            )
                            import traceback
                            traceback.print_exc()
        
        # Save summary
        self._save_summary()
        
        return self.results
    
    def _save_result(self, result: ConversationResult):
        """Save a single result to disk."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        filename = f"{result.scenario_id}_{result.persona_id}_{result.strategy}_run{result.run_id}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "scenario_id": result.scenario_id,
                "persona_id": result.persona_id,
                "strategy": result.strategy,
                "run_id": result.run_id,
                "turns": result.turns,
                "metrics": {
                    "total_turns": result.total_turns,
                    "goals_detected": result.goals_detected,
                    "full_context_tokens": result.full_context_tokens,
                    "compressed_context_tokens": result.compressed_context_tokens,
                    "token_reduction_pct": result.token_reduction_pct,
                    "tool_calls_total": result.tool_calls_total,
                    "tool_calls_successful": result.tool_calls_successful,
                    "tool_accuracy": result.tool_accuracy,
                    "tools_used": result.tools_used
                },
                "timing": {
                    "started_at": result.started_at,
                    "completed_at": result.completed_at,
                    "duration_seconds": result.duration_seconds
                }
            }, f, indent=2)
        
        logger.debug(f"Saved result to {filepath}")
    
    def _save_summary(self):
        """Save experiment summary."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, "summary.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        logger.info(f"Saved summary to {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all results."""
        if not self.results:
            return {"error": "No results yet"}
        
        by_strategy = {}
        for result in self.results:
            if result.strategy not in by_strategy:
                by_strategy[result.strategy] = []
            by_strategy[result.strategy].append(result)
        
        summary = {
            "total_runs": len(self.results),
            "by_strategy": {}
        }
        
        for strategy, results in by_strategy.items():
            n = len(results)
            summary["by_strategy"][strategy] = {
                "runs": n,
                "avg_turns": sum(r.total_turns for r in results) / n,
                "avg_token_reduction": sum(r.token_reduction_pct for r in results) / n,
                "avg_tool_accuracy": sum(r.tool_accuracy for r in results) / n,
                "total_tool_calls": sum(r.tool_calls_total for r in results),
                "successful_tool_calls": sum(r.tool_calls_successful for r in results),
                "avg_duration": sum(r.duration_seconds for r in results) / n
            }
        
        return summary
    
    def print_report(self):
        """Print a formatted report of results."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("CONTEXT COMPACTION EXPERIMENT RESULTS")
        print("="*70)
        print(f"\nTotal runs: {summary['total_runs']}")
        
        print("\n" + "-"*70)
        print(f"{'Strategy':<20} | {'Token Reduction':>15} | {'Tool Accuracy':>13} | {'Avg Turns':>10}")
        print("-"*70)
        
        for strategy, stats in summary["by_strategy"].items():
            print(f"{strategy:<20} | {stats['avg_token_reduction']:>14.1f}% | {stats['avg_tool_accuracy']*100:>12.1f}% | {stats['avg_turns']:>10.1f}")
        
        print("-"*70)
        print()
