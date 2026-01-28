#!/usr/bin/env python3
"""
Fair Replay Mode Comparison - Ensures identical user messages across all strategies.

Phase 1: Generate baseline conversation with simulated user
Phase 2: Replay SAME user messages against each strategy
Metrics: Token Savings, Tool Accuracy, GSR, Latency
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

from agents.bedrock_client import BedrockClient
from agents.judge_agent import JudgeAgent
from core.models import Turn, Session, TurnQuality

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

GOALS = [
    "Find the current weather in San Francisco and what to wear",
    "Calculate how much I'll save if I invest $5000 at 7% interest for 5 years"
]

USER_PERSONA = """You are a practical user who asks clear questions.
You have two goals to accomplish. After each goal is addressed, 
acknowledge success and move to the next one.
When done, thank the assistant and end the conversation."""

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

MAX_TURNS = 6


@dataclass 
class ReplayResult:
    """Results from a replay run."""
    strategy: str
    turns: int
    token_reduction_pct: float
    tool_calls: int
    tool_success: int
    tool_accuracy: float
    gsr: float
    turn_success_rate: float
    latency_seconds: float
    conversation: List[Dict[str, Any]]
    evaluations: List[Dict[str, Any]] = field(default_factory=list)


async def generate_baseline_conversation() -> List[str]:
    """
    Phase 1: Generate baseline conversation with simulated user.
    Returns list of user messages to replay.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating Baseline User Messages")
    logger.info("=" * 60)
    
    user = SimulatedUser(
        goals=GOALS.copy(),
        persona=USER_PERSONA
    )
    
    # Create a simple chatbot just to drive the conversation
    tools = ToolRouter()
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools.get_tools_description())
    chatbot = ChatbotAgent(
        context_strategy=FullContextStrategy(),
        tool_router=tools,
        system_prompt=system_prompt
    )
    
    user_messages = []
    
    # Get initial message
    msg = await user.generate_utterance()
    user_messages.append(msg)
    logger.info(f"  User[1]: {msg[:80]}...")
    
    for turn in range(MAX_TURNS - 1):
        # Get agent response
        response = await chatbot.respond(msg, is_new_goal=(turn == 0))
        logger.info(f"  Agent[{turn+1}]: {response[:80]}...")
        
        # Check if done
        if user.all_goals_achieved:
            break
            
        # Get next user message
        msg = await user.generate_utterance(response)
        user_messages.append(msg)
        logger.info(f"  User[{turn+2}]: {msg[:80]}...")
    
    logger.info(f"\nGenerated {len(user_messages)} user messages for replay")
    return user_messages


async def replay_with_strategy(
    strategy_name: str,
    user_messages: List[str],
    judge: JudgeAgent
) -> ReplayResult:
    """
    Phase 2: Replay user messages with a specific strategy.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"REPLAY: {strategy_name}")
    logger.info(f"{'='*60}")
    
    # Create fresh tools
    tools = ToolRouter()
    
    # Create strategy
    if strategy_name == "full":
        strategy = FullContextStrategy()
    elif strategy_name == "sliding_window":
        strategy = SlidingWindowStrategy(window_size=4)
    elif strategy_name == "goal_boundary":
        summarizer = GoalSummarizer()
        strategy = GoalBoundaryStrategy(summarizer=summarizer)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Create chatbot
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools.get_tools_description())
    chatbot = ChatbotAgent(
        context_strategy=strategy,
        tool_router=tools,
        system_prompt=system_prompt
    )
    
    goal_detector = GoalDetector()
    
    # Track conversation
    conversation = []
    start_time = datetime.now()
    
    # Replay each user message
    for i, user_msg in enumerate(user_messages):
        turn_num = i + 1
        
        # Detect if new goal (based on previous messages)
        if i == 0:
            is_new_goal = True
        else:
            is_new_goal = await goal_detector.is_new_goal(
                user_msg,
                [c["user"] for c in conversation]
            )
        
        if is_new_goal and i > 0:
            logger.info(f"  >>> New goal detected at turn {turn_num}")
        
        logger.info(f"  Turn {turn_num}: User: {user_msg[:60]}...")
        
        # Get chatbot response
        agent_response = await chatbot.respond(user_msg, is_new_goal=is_new_goal)
        logger.info(f"  Turn {turn_num}: Agent: {agent_response[:60]}...")
        
        conversation.append({
            "turn": turn_num,
            "user": user_msg,
            "agent": agent_response,
            "is_new_goal": is_new_goal
        })
    
    latency = (datetime.now() - start_time).total_seconds()
    
    # Get metrics
    metrics = chatbot.get_metrics()
    tool_metrics = metrics.get("tool_metrics", {})
    
    # Evaluate with Judge
    logger.info(f"  Evaluating with Judge...")
    eval_result = await evaluate_with_judge(conversation, judge)
    
    return ReplayResult(
        strategy=strategy_name,
        turns=len(conversation),
        token_reduction_pct=metrics.get("token_reduction_pct", 0),
        tool_calls=tool_metrics.get("total_calls", 0),
        tool_success=tool_metrics.get("successful_calls", 0),
        tool_accuracy=tool_metrics.get("accuracy", 0) * 100,
        gsr=eval_result["gsr"],
        turn_success_rate=eval_result["turn_success_rate"],
        latency_seconds=latency,
        conversation=conversation,
        evaluations=eval_result["evaluations"]
    )


async def evaluate_with_judge(
    conversation: List[Dict[str, Any]],
    judge: JudgeAgent
) -> Dict[str, Any]:
    """Evaluate conversation using Judge Agent for GSR."""
    
    turns = [
        Turn(turn_number=c["turn"], user_message=c["user"], agent_response=c["agent"])
        for c in conversation
    ]
    
    session = Session(session_id="replay", turns=turns, goals=[])
    evaluations = await judge.evaluate_session(session)
    
    # Calculate GSR
    goals = []
    current_goal_turns = []
    
    for eval_result in evaluations:
        if eval_result.is_new_goal and current_goal_turns:
            goal_success = all(t.quality == TurnQuality.SUCCESS for t in current_goal_turns)
            goals.append(goal_success)
            current_goal_turns = []
        current_goal_turns.append(eval_result)
    
    if current_goal_turns:
        goal_success = all(t.quality == TurnQuality.SUCCESS for t in current_goal_turns)
        goals.append(goal_success)
    
    gsr = sum(goals) / len(goals) * 100 if goals else 0
    turn_sr = sum(1 for e in evaluations if e.quality == TurnQuality.SUCCESS) / len(evaluations) * 100 if evaluations else 0
    
    return {
        "gsr": gsr,
        "turn_success_rate": turn_sr,
        "evaluations": [
            {
                "turn": e.turn_number,
                "is_new_goal": e.is_new_goal,
                "quality": e.quality.value,
                "rcof": e.rcof,
                "reasoning": e.reasoning[:100] if e.reasoning else ""
            }
            for e in evaluations
        ]
    }


async def main():
    """Run fair replay comparison."""
    
    print("\n" + "=" * 70)
    print("FAIR REPLAY COMPARISON")
    print("Same user messages across all strategies")
    print("=" * 70)
    print(f"\nGoals: {GOALS}")
    print(f"Max Turns: {MAX_TURNS}")
    print("=" * 70 + "\n")
    
    # Phase 1: Generate baseline
    user_messages = await generate_baseline_conversation()
    
    print(f"\n{'='*70}")
    print(f"USER MESSAGES TO REPLAY ({len(user_messages)} messages)")
    print(f"{'='*70}")
    for i, msg in enumerate(user_messages):
        print(f"  [{i+1}] {msg[:100]}...")
    
    # Wait before Phase 2
    print("\nWaiting 10s before replay phase...")
    await asyncio.sleep(10)
    
    # Phase 2: Replay with each strategy
    judge = JudgeAgent()
    strategies = ["full", "sliding_window", "goal_boundary"]
    results: List[ReplayResult] = []
    
    for strategy in strategies:
        try:
            result = await replay_with_strategy(strategy, user_messages, judge)
            results.append(result)
            logger.info(f"\n✓ {strategy}: {result.turns} turns, "
                       f"{result.token_reduction_pct:.1f}% reduction, "
                       f"GSR={result.gsr:.0f}%")
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"✗ {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    print("\n" + "=" * 80)
    print("FAIR COMPARISON RESULTS (Same User Messages)")
    print("=" * 80)
    
    header = f"{'Strategy':<20} | {'Turns':>5} | {'Token Red':>9} | {'Tool Acc':>8} | {'GSR':>5} | {'Latency':>8}"
    print(header)
    print("-" * 80)
    
    for r in results:
        print(f"{r.strategy:<20} | {r.turns:>5} | {r.token_reduction_pct:>8.1f}% | "
              f"{r.tool_accuracy:>7.0f}% | {r.gsr:>4.0f}% | {r.latency_seconds:>7.1f}s")
    
    print("-" * 80)
    
    # Save results
    output_dir = Path("context_eng_expt/results")
    output_dir.mkdir(exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "type": "fair_replay_comparison",
        "config": {
            "goals": GOALS,
            "max_turns": MAX_TURNS,
            "user_persona": USER_PERSONA[:100] + "..."
        },
        "user_messages": user_messages,
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
                "latency_seconds": r.latency_seconds,
                "conversation": r.conversation,
                "evaluations": r.evaluations
            }
            for r in results
        ]
    }
    
    output_file = output_dir / "fair_replay_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
