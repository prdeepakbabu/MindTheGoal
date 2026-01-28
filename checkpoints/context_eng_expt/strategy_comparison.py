#!/usr/bin/env python3
"""Compare all 3 strategies on the same scenario."""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_eng_expt.config import ExperimentConfig, CompactionStrategy
from context_eng_expt.experiment.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def run_comparison():
    """Run all 3 strategies and compare results."""
    
    print("\n" + "="*70)
    print("STRATEGY COMPARISON TEST")
    print("="*70)
    
    config = ExperimentConfig(
        max_turns_per_conversation=6,  # Short conversations
        num_runs_per_scenario=1,
        output_dir="context_eng_expt/results",
        save_conversations=True,
        enable_tools=True
    )
    
    runner = ExperimentRunner(config)
    
    strategies = [
        CompactionStrategy.FULL_CONTEXT,
        CompactionStrategy.SLIDING_WINDOW,
        CompactionStrategy.GOAL_BOUNDARY,
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n>>> Running {strategy.value} <<<")
        try:
            result = await runner.run_single(
                scenario_id="research_and_plan",
                persona_id="efficient",
                strategy=strategy,
                run_id=1
            )
            results.append({
                "strategy": strategy.value,
                "turns": result.total_turns,
                "token_reduction_pct": result.token_reduction_pct,
                "tool_calls": result.tool_calls_total,
                "tool_success": result.tool_calls_successful,
                "tool_accuracy": result.tool_accuracy,
                "duration": result.duration_seconds,
                "status": "success"
            })
            print(f"✓ {strategy.value}: {result.total_turns} turns, {result.token_reduction_pct:.1f}% reduction, {result.tool_accuracy*100:.0f}% tool accuracy")
        except Exception as e:
            results.append({
                "strategy": strategy.value,
                "error": str(e),
                "status": "failed"
            })
            print(f"✗ {strategy.value} failed: {e}")
        
        # Wait between runs
        await asyncio.sleep(5)
    
    # Print comparison matrix
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Strategy':<20} | {'Turns':>6} | {'Token Reduction':>15} | {'Tool Accuracy':>13} | {'Duration':>10}")
    print("-"*70)
    
    for r in results:
        if r["status"] == "success":
            print(f"{r['strategy']:<20} | {r['turns']:>6} | {r['token_reduction_pct']:>14.1f}% | {r['tool_accuracy']*100:>12.0f}% | {r['duration']:>9.1f}s")
        else:
            print(f"{r['strategy']:<20} | {'FAILED':>6} | {'-':>15} | {'-':>13} | {'-':>10}")
    
    print("-"*70)
    
    # Save results
    output_file = "context_eng_expt/results/comparison_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "scenario": "research_and_plan",
            "persona": "efficient",
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_comparison())
