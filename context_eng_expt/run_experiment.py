#!/usr/bin/env python3
"""
CLI entry point for running context compaction experiments.

Usage:
    # Run a single experiment
    python -m context_eng_expt.run_experiment --scenario single_goal_short --strategy goal_boundary
    
    # Run quick test (one scenario, all strategies)
    python -m context_eng_expt.run_experiment --quick-test
    
    # Run full experiment matrix
    python -m context_eng_expt.run_experiment --full-matrix
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from .config import ExperimentConfig, CompactionStrategy, SCENARIOS, PERSONAS
from .experiment.runner import ExperimentRunner
from .experiment.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run context compaction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with specific configuration
  python -m context_eng_expt.run_experiment --scenario multi_goal_sequential --strategy goal_boundary --persona efficient
  
  # Quick test - one scenario, all strategies, one persona
  python -m context_eng_expt.run_experiment --quick-test
  
  # Full matrix - all scenarios, all strategies, all personas
  python -m context_eng_expt.run_experiment --full-matrix
  
  # Compare two strategies
  python -m context_eng_expt.run_experiment --compare full goal_boundary --scenario multi_goal_complex
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--quick-test',
        action='store_true',
        help='Run a quick test with one scenario and all strategies'
    )
    mode_group.add_argument(
        '--full-matrix',
        action='store_true',
        help='Run the full experiment matrix (all combinations)'
    )
    mode_group.add_argument(
        '--compare',
        nargs=2,
        metavar=('STRATEGY1', 'STRATEGY2'),
        help='Compare two strategies on specified scenarios'
    )
    
    # Specific configuration
    parser.add_argument(
        '--scenario',
        choices=[s['id'] for s in SCENARIOS],
        help='Scenario to run'
    )
    parser.add_argument(
        '--strategy',
        choices=[s.value for s in CompactionStrategy],
        help='Compaction strategy to use'
    )
    parser.add_argument(
        '--persona',
        choices=[p['id'] for p in PERSONAS],
        default='efficient',
        help='User persona (default: efficient)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per configuration (default: 1)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='context_eng_expt/results',
        help='Directory for output files'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


async def run_single(args):
    """Run a single experiment configuration."""
    if not args.scenario or not args.strategy:
        print("Error: --scenario and --strategy are required for single run")
        print("Use --quick-test or --full-matrix for automated runs")
        sys.exit(1)
    
    config = ExperimentConfig(
        strategy=CompactionStrategy(args.strategy),
        num_runs_per_scenario=args.runs,
        output_dir=args.output_dir,
        save_conversations=not args.no_save
    )
    
    runner = ExperimentRunner(config)
    
    print(f"\n{'='*60}")
    print(f"Running: {args.scenario} / {args.persona} / {args.strategy}")
    print(f"{'='*60}\n")
    
    for run_id in range(1, args.runs + 1):
        result = await runner.run_single(
            scenario_id=args.scenario,
            persona_id=args.persona,
            strategy=CompactionStrategy(args.strategy),
            run_id=run_id
        )
        
        print(f"\nRun {run_id} completed:")
        print(f"  Turns: {result.total_turns}")
        print(f"  Goals detected: {result.goals_detected}")
        print(f"  Token reduction: {result.token_reduction_pct:.1f}%")
        print(f"  Duration: {result.duration_seconds:.1f}s")
    
    return runner.get_summary()


async def run_quick_test(args):
    """Run a quick test with one scenario and all strategies."""
    config = ExperimentConfig(
        num_runs_per_scenario=1,
        output_dir=args.output_dir,
        save_conversations=not args.no_save
    )
    
    runner = ExperimentRunner(config)
    
    scenario = args.scenario or "single_goal_short"
    persona = args.persona or "efficient"
    
    print(f"\n{'='*60}")
    print("QUICK TEST")
    print(f"Scenario: {scenario}")
    print(f"Persona: {persona}")
    print(f"Strategies: {[s.value for s in CompactionStrategy]}")
    print(f"{'='*60}\n")
    
    for strategy in CompactionStrategy:
        print(f"\n--- Testing {strategy.value} ---")
        try:
            result = await runner.run_single(
                scenario_id=scenario,
                persona_id=persona,
                strategy=strategy,
                run_id=1
            )
            print(f"  ✓ Turns: {result.total_turns}, Token reduction: {result.token_reduction_pct:.1f}%")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return runner.get_summary()


async def run_full_matrix(args):
    """Run the full experiment matrix."""
    config = ExperimentConfig(
        num_runs_per_scenario=args.runs,
        output_dir=args.output_dir,
        save_conversations=not args.no_save
    )
    
    runner = ExperimentRunner(config)
    
    total = len(SCENARIOS) * len(PERSONAS) * len(CompactionStrategy) * args.runs
    
    print(f"\n{'='*60}")
    print("FULL EXPERIMENT MATRIX")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Personas: {len(PERSONAS)}")
    print(f"Strategies: {len(CompactionStrategy)}")
    print(f"Runs per config: {args.runs}")
    print(f"Total runs: {total}")
    print(f"{'='*60}\n")
    
    confirm = input("This may take a while and use significant API calls. Continue? [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return None
    
    results = await runner.run_matrix()
    
    print(f"\n\nCompleted {len(results)} runs")
    return runner.get_summary()


async def run_compare(args):
    """Compare two strategies."""
    strategy1 = CompactionStrategy(args.compare[0])
    strategy2 = CompactionStrategy(args.compare[1])
    
    config = ExperimentConfig(
        num_runs_per_scenario=args.runs,
        output_dir=args.output_dir,
        save_conversations=not args.no_save
    )
    
    runner = ExperimentRunner(config)
    
    scenarios = [args.scenario] if args.scenario else [s['id'] for s in SCENARIOS]
    persona = args.persona
    
    print(f"\n{'='*60}")
    print(f"COMPARING: {strategy1.value} vs {strategy2.value}")
    print(f"Scenarios: {scenarios}")
    print(f"Persona: {persona}")
    print(f"{'='*60}\n")
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario} ---")
        
        for strategy in [strategy1, strategy2]:
            result = await runner.run_single(
                scenario_id=scenario,
                persona_id=persona,
                strategy=strategy,
                run_id=1
            )
            print(f"  {strategy.value}: {result.token_reduction_pct:.1f}% reduction, {result.total_turns} turns")
    
    return runner.get_summary()


async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = datetime.now()
    
    try:
        if args.quick_test:
            summary = await run_quick_test(args)
        elif args.full_matrix:
            summary = await run_full_matrix(args)
        elif args.compare:
            summary = await run_compare(args)
        else:
            summary = await run_single(args)
        
        if summary:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            
            if "by_strategy" in summary:
                for strategy, stats in summary["by_strategy"].items():
                    print(f"\n{strategy}:")
                    print(f"  Runs: {stats['runs']}")
                    print(f"  Avg turns: {stats['avg_turns']:.1f}")
                    print(f"  Avg token reduction: {stats['avg_token_reduction']:.1f}%")
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n\nTotal time: {duration:.1f}s")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
