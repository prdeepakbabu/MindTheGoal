#!/usr/bin/env python3
"""Minimal smoke test - single conversation with full context strategy."""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_eng_expt.config import ExperimentConfig, CompactionStrategy
from context_eng_expt.experiment.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def smoke_test():
    """Run a single conversation to verify everything works."""
    
    print("\n" + "="*60)
    print("SMOKE TEST - Single Conversation")
    print("="*60)
    
    config = ExperimentConfig(
        max_turns_per_conversation=5,  # Very short
        num_runs_per_scenario=1,
        output_dir="context_eng_expt/results",
        save_conversations=True,
        enable_tools=True
    )
    
    runner = ExperimentRunner(config)
    
    try:
        result = await runner.run_single(
            scenario_id="research_and_plan",
            persona_id="efficient",
            strategy=CompactionStrategy.FULL_CONTEXT,
            run_id=1
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Total turns: {result.total_turns}")
        print(f"Token reduction: {result.token_reduction_pct:.1f}%")
        print(f"Tool calls: {result.tool_calls_total}")
        print(f"Tool accuracy: {result.tool_accuracy*100:.0f}%")
        print(f"Duration: {result.duration_seconds:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(smoke_test())
