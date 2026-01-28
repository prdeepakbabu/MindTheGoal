#!/usr/bin/env python3
"""Quick test script for the context compaction experiment."""

import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_eng_expt.config import ExperimentConfig, CompactionStrategy, SCENARIOS
from context_eng_expt.experiment.runner import ExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_quick_test():
    """Run a quick test with one scenario and all strategies."""
    
    config = ExperimentConfig(
        max_turns_per_conversation=15,
        num_runs_per_scenario=1,
        output_dir="context_eng_expt/results",
        save_conversations=True,
        enable_tools=True
    )
    
    runner = ExperimentRunner(config)
    
    # Test with first scenario, first persona, all strategies
    scenario_id = "research_and_plan"
    persona_id = "efficient"
    
    strategies = [
        CompactionStrategy.FULL_CONTEXT,
        CompactionStrategy.SLIDING_WINDOW,
        CompactionStrategy.GOAL_BOUNDARY,
    ]
    
    print("\n" + "="*70)
    print("CONTEXT COMPACTION EXPERIMENT - QUICK TEST")
    print("="*70)
    print(f"Scenario: {scenario_id}")
    print(f"Persona: {persona_id}")
    print(f"Strategies: {[s.value for s in strategies]}")
    print("="*70 + "\n")
    
    for strategy in strategies:
        try:
            print(f"\n>>> Testing {strategy.value} <<<")
            result = await runner.run_single(
                scenario_id=scenario_id,
                persona_id=persona_id,
                strategy=strategy,
                run_id=1
            )
            print(f"✓ Completed: {result.total_turns} turns, {result.token_reduction_pct:.1f}% token reduction")
            if result.tool_calls_total > 0:
                print(f"  Tools: {result.tool_calls_successful}/{result.tool_calls_total} successful ({result.tool_accuracy*100:.0f}%)")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait between strategies (API rate limiting)
        await asyncio.sleep(2)
    
    # Print summary
    runner.print_report()
    
    return runner.get_summary()


if __name__ == "__main__":
    asyncio.run(run_quick_test())
