#!/usr/bin/env python3
"""
Fair comparison experiment with consistent prompts and GSR evaluation.

This script ensures:
1. Identical system prompts across all 3 strategies
2. Same user persona and goals
3. GSR evaluation using Judge Agent
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, '.')

from agents.bedrock_client import BedrockClient
from agents.judge_agent import JudgeAgent
from core.models import Turn, Session, TurnQuality, TurnEvaluation

from context_eng_expt.agents.chatbot_agent import ChatbotAgent
from context_eng_expt.agents.simulated_user import SimulatedUser
from context_eng_expt.agents.goal_detector import GoalDetector
from context_eng_expt.context.strategies import (
    FullContextStrategy,
    SlidingWindowStrategy,
    GoalBoundaryStrategy,
)
from context_eng_expt.context.goal_summarizer import GoalSummarizer
from context_eng_expt.tools.router import ToolRouter
from context_eng_expt.tools.search import WebSearchTool
from context_eng_expt.tools.scraper import WebScraperTool
from context_eng_expt.tools.code_executor import CodeExecutorTool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# EXPERIMENT CONFIGURATION - SAME FOR ALL STRATEGIES
# ============================================================

GOALS = [
    "Find information about the top 3 programming languages in 2024",
    "Calculate compound interest for a $10,000 investment at 5% for 10 years"
]

USER_PERSONA = """You are a user who asks clear, direct questions. 
You have two specific goals to accomplish in this conversation.
After achieving each goal, acknowledge it and move to the next.
When both goals are done, thank the assistant and end the conversation."""

# System prompt - IDENTICAL for all strategies
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

## Available Tools

{tools_description}

## Tool Usage Format

When you need to use a tool, respond with:
<tool_use>
<name>tool_name</name>
<arguments>{{"param1": "value1"}}</arguments>
</tool_use>

## Guidelines

- Use tools when needed for accurate information
- Be helpful and concise
- If a tool fails, explain and try alternatives"""

MAX_TURNS = 8


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    strategy: str
    turns: int
    token_reduction_pct: float
    tool_calls: int
    tool_success: int
    tool_accuracy: float
    gsr: float  # Goal Success Rate from Judge
    turn_success_rate: float
    rcof_distribution: Dict[str, int]
    conversation: List[Dict[str, Any]]
    evaluations: List[Dict[str, Any]]
    duration_seconds: float


def create_tools() -> ToolRouter:
    """Create tool router with default tools (auto-registered)."""
    return ToolRouter()  # Default tools are registered automatically


async def run_conversation(
    strategy_name: str,
    context_strategy,
    tools: ToolRouter
) -> Dict[str, Any]:
    """Run a single conversation with the given strategy."""
    
    # Create agents with IDENTICAL configuration
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools.get_tools_description())
    
    chatbot = ChatbotAgent(
        context_strategy=context_strategy,
        tool_router=tools,
        system_prompt=system_prompt,
        max_tool_iterations=3
    )
    
    user = SimulatedUser(
        goals=GOALS.copy(),
        persona=USER_PERSONA
    )
    
    goal_detector = GoalDetector()
    
    # Track conversation
    conversation = []
    start_time = datetime.now()
    
    # Initial user message
    user_message = await user.generate_utterance()
    is_new_goal = True
    
    for turn_num in range(1, MAX_TURNS + 1):
        logger.info(f"  Turn {turn_num}: User: {user_message[:60]}...")
        
        # Get chatbot response
        agent_response = await chatbot.respond(user_message, is_new_goal=is_new_goal)
        logger.info(f"  Turn {turn_num}: Agent: {agent_response[:60]}...")
        
        # Record turn
        conversation.append({
            "turn": turn_num,
            "user": user_message,
            "agent": agent_response,
            "is_new_goal": is_new_goal
        })
        
        # Check if conversation should end
        if user.all_goals_achieved or turn_num >= MAX_TURNS:
            break
        
        # Generate next user message
        user_message = await user.generate_utterance(agent_response)
        
        # Detect if new goal
        is_new_goal = await goal_detector.is_new_goal(
            user_message, 
            [t["user"] for t in conversation]
        )
        
        if is_new_goal:
            logger.info(f"  >>> New goal detected!")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    # Get metrics
    metrics = chatbot.get_metrics()
    
    return {
        "conversation": conversation,
        "metrics": metrics,
        "duration": duration,
        "strategy": strategy_name
    }


async def evaluate_with_judge(
    conversation: List[Dict[str, Any]],
    judge: JudgeAgent
) -> Dict[str, Any]:
    """Evaluate conversation using Judge Agent for GSR."""
    
    # Convert to Turn objects
    turns = []
    for c in conversation:
        turns.append(Turn(
            turn_number=c["turn"],
            user_message=c["user"],
            agent_response=c["agent"]
        ))
    
    # Create session
    session = Session(
        session_id="evaluation",
        turns=turns,
        goals=[]
    )
    
    # Evaluate
    evaluations = await judge.evaluate_session(session)
    
    # Calculate GSR
    # Per paper: Goal is successful IFF all turns within it are successful
    goals = []
    current_goal_turns = []
    
    for eval_result in evaluations:
        if eval_result.is_new_goal and current_goal_turns:
            # Save previous goal
            goal_success = all(t.quality == TurnQuality.SUCCESS for t in current_goal_turns)
            goals.append(goal_success)
            current_goal_turns = []
        current_goal_turns.append(eval_result)
    
    # Don't forget last goal
    if current_goal_turns:
        goal_success = all(t.quality == TurnQuality.SUCCESS for t in current_goal_turns)
        goals.append(goal_success)
    
    gsr = sum(goals) / len(goals) * 100 if goals else 0
    
    # Calculate turn success rate
    successful_turns = sum(1 for e in evaluations if e.quality == TurnQuality.SUCCESS)
    turn_success_rate = successful_turns / len(evaluations) * 100 if evaluations else 0
    
    # Count RCOF
    rcof_counts = {}
    for e in evaluations:
        if e.rcof:
            rcof_counts[e.rcof] = rcof_counts.get(e.rcof, 0) + 1
    
    return {
        "gsr": gsr,
        "turn_success_rate": turn_success_rate,
        "goals_detected": len(goals),
        "goals_successful": sum(goals),
        "rcof_distribution": rcof_counts,
        "evaluations": [
            {
                "turn": e.turn_number,
                "is_new_goal": e.is_new_goal,
                "quality": e.quality.value,
                "rcof": e.rcof,
                "reasoning": e.reasoning
            }
            for e in evaluations
        ]
    }


async def run_strategy(strategy_name: str, judge: JudgeAgent) -> ExperimentResult:
    """Run experiment with a specific strategy."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {strategy_name}")
    logger.info(f"{'='*60}")
    
    # Create fresh tools
    tools = create_tools()
    
    # Create strategy
    if strategy_name == "full":
        strategy = FullContextStrategy()
    elif strategy_name == "sliding_window":
        strategy = SlidingWindowStrategy(window_size=6)
    elif strategy_name == "goal_boundary":
        summarizer = GoalSummarizer()
        strategy = GoalBoundaryStrategy(summarizer=summarizer)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Run conversation
    result = await run_conversation(strategy_name, strategy, tools)
    
    # Evaluate with Judge for GSR
    logger.info(f"  Evaluating with Judge Agent...")
    eval_result = await evaluate_with_judge(result["conversation"], judge)
    
    # Build final result
    metrics = result["metrics"]
    tool_metrics = metrics.get("tool_metrics", {})
    
    return ExperimentResult(
        strategy=strategy_name,
        turns=len(result["conversation"]),
        token_reduction_pct=metrics.get("token_reduction_pct", 0),
        tool_calls=tool_metrics.get("total_calls", 0),
        tool_success=tool_metrics.get("successful_calls", 0),
        tool_accuracy=tool_metrics.get("accuracy", 0) * 100,
        gsr=eval_result["gsr"],
        turn_success_rate=eval_result["turn_success_rate"],
        rcof_distribution=eval_result["rcof_distribution"],
        conversation=result["conversation"],
        evaluations=eval_result["evaluations"],
        duration_seconds=result["duration"]
    )


async def main():
    """Run fair comparison experiment with GSR."""
    
    print("\n" + "=" * 70)
    print("FAIR COMPARISON WITH GSR EVALUATION")
    print("=" * 70)
    print(f"\nGoals: {GOALS}")
    print(f"Max Turns: {MAX_TURNS}")
    print("=" * 70 + "\n")
    
    # Create shared Judge Agent
    judge = JudgeAgent()
    
    # Run all strategies
    strategies = ["full", "sliding_window", "goal_boundary"]
    results: List[ExperimentResult] = []
    
    for strategy in strategies:
        try:
            result = await run_strategy(strategy, judge)
            results.append(result)
            logger.info(f"\n✓ {strategy}: {result.turns} turns, "
                       f"{result.token_reduction_pct:.1f}% reduction, "
                       f"GSR={result.gsr:.0f}%")
            
            # Wait between runs for API rate limits
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"✗ {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    header = f"{'Strategy':<20} | {'Turns':>5} | {'Token Red':>9} | {'Tool Acc':>8} | {'GSR':>5} | {'Turn SR':>7}"
    print(header)
    print("-" * 70)
    
    for r in results:
        print(f"{r.strategy:<20} | {r.turns:>5} | {r.token_reduction_pct:>8.1f}% | "
              f"{r.tool_accuracy:>7.0f}% | {r.gsr:>4.0f}% | {r.turn_success_rate:>6.0f}%")
    
    print("-" * 70)
    
    # Save detailed results
    output_dir = Path("context_eng_expt/results")
    output_dir.mkdir(exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "goals": GOALS,
            "max_turns": MAX_TURNS,
            "user_persona": USER_PERSONA[:100] + "..."
        },
        "results": [
            {
                "strategy": r.strategy,
                "turns": r.turns,
                "token_reduction_pct": r.token_reduction_pct,
                "tool_calls": r.tool_calls,
                "tool_success": r.tool_success,
                "tool_accuracy": r.tool_accuracy,
                "gsr": r.gsr,
                "turn_success_rate": r.turn_success_rate,
                "rcof_distribution": r.rcof_distribution,
                "duration_seconds": r.duration_seconds,
                "conversation": r.conversation,
                "evaluations": r.evaluations
            }
            for r in results
        ]
    }
    
    output_file = output_dir / "gsr_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
