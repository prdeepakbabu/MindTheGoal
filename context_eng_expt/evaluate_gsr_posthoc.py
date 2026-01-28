#!/usr/bin/env python3
"""
Post-hoc GSR evaluation of existing experiment results.
Runs the Judge Agent on saved conversations to compute GSR.
"""

import asyncio
import json
from pathlib import Path

import sys
sys.path.insert(0, '.')

from agents.judge_agent import JudgeAgent
from core.models import Turn, Session, TurnQuality


async def evaluate_conversation(judge: JudgeAgent, conversation_file: Path):
    """Load and evaluate a conversation from file."""
    
    with open(conversation_file) as f:
        data = json.load(f)
    
    strategy = data.get("strategy", conversation_file.stem.split("_")[-2])
    
    # Extract turns
    turns_data = data.get("turns", [])
    
    turns = []
    for t in turns_data:
        turns.append(Turn(
            turn_number=t["turn"],
            user_message=t["user"],
            agent_response=t["agent"]
        ))
    
    if not turns:
        return None
    
    # Create session and evaluate
    session = Session(session_id=conversation_file.stem, turns=turns, goals=[])
    evaluations = await judge.evaluate_session(session)
    
    # Calculate GSR (goal success = all turns in goal successful)
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
        "strategy": strategy,
        "file": conversation_file.name,
        "turns": len(turns),
        "goals_detected": len(goals),
        "goals_successful": sum(goals),
        "gsr": gsr,
        "turn_success_rate": turn_sr,
        "evaluations": [
            {
                "turn": e.turn_number,
                "is_new_goal": e.is_new_goal,
                "quality": e.quality.value,
                "rcof": e.rcof,
                "reasoning": e.reasoning[:100] + "..." if len(e.reasoning) > 100 else e.reasoning
            }
            for e in evaluations
        ]
    }


async def main():
    """Evaluate all existing conversations for GSR."""
    
    results_dir = Path("context_eng_expt/results")
    
    # Find conversation files
    conversation_files = list(results_dir.glob("research_and_plan_efficient_*_run1.json"))
    
    print("=" * 70)
    print("POST-HOC GSR EVALUATION")
    print("=" * 70)
    print(f"\nFound {len(conversation_files)} conversation files")
    
    judge = JudgeAgent()
    results = []
    
    for conv_file in conversation_files:
        print(f"\nEvaluating: {conv_file.name}")
        try:
            result = await evaluate_conversation(judge, conv_file)
            if result:
                results.append(result)
                print(f"  GSR: {result['gsr']:.0f}% | Turn SR: {result['turn_success_rate']:.0f}% | Goals: {result['goals_detected']}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("GSR COMPARISON RESULTS")
    print("=" * 70)
    
    header = f"{'Strategy':<20} | {'Turns':>5} | {'Goals':>5} | {'GSR':>6} | {'Turn SR':>7}"
    print(header)
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x["strategy"]):
        strategy = r["strategy"].replace("research_and_plan_efficient_", "").replace("_run1", "")
        print(f"{strategy:<20} | {r['turns']:>5} | {r['goals_detected']:>5} | {r['gsr']:>5.0f}% | {r['turn_success_rate']:>6.0f}%")
    
    print("-" * 70)
    
    # Save results
    output = {
        "type": "posthoc_gsr_evaluation",
        "results": results
    }
    
    output_file = results_dir / "gsr_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
