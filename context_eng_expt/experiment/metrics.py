"""Metrics collection and analysis for the context compaction experiment."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""
    
    # Identifiers
    scenario_id: str
    persona_id: str
    strategy: str
    run_id: int
    
    # Token metrics
    total_input_tokens: int = 0
    tokens_if_full_context: int = 0
    token_reduction_pct: float = 0.0
    
    # Conversation metrics
    total_turns: int = 0
    total_goals: int = 0
    goals_achieved: int = 0
    
    # Quality metrics (from evaluation)
    gsr: float = 0.0  # Goal Success Rate
    turn_success_rate: float = 0.0
    
    # Tool metrics
    tool_calls_total: int = 0
    tool_calls_successful: int = 0
    tool_accuracy: float = 0.0
    tools_used: Dict[str, int] = field(default_factory=dict)
    
    # Efficiency metrics
    avg_latency_ms: float = 0.0
    total_duration_seconds: float = 0.0
    
    # Compression metrics
    goal_summaries: List[str] = field(default_factory=list)
    compression_ratios: List[float] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates metrics across experiment runs."""
    
    def __init__(self):
        self.metrics: List[ExperimentMetrics] = []
    
    def add(self, metrics: ExperimentMetrics):
        """Add metrics from a single run."""
        self.metrics.append(metrics)
    
    def aggregate_by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics grouped by strategy."""
        by_strategy: Dict[str, List[ExperimentMetrics]] = {}
        
        for m in self.metrics:
            if m.strategy not in by_strategy:
                by_strategy[m.strategy] = []
            by_strategy[m.strategy].append(m)
        
        aggregated = {}
        for strategy, metrics_list in by_strategy.items():
            aggregated[strategy] = self._compute_aggregates(metrics_list)
        
        return aggregated
    
    def aggregate_by_scenario(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics grouped by scenario."""
        by_scenario: Dict[str, List[ExperimentMetrics]] = {}
        
        for m in self.metrics:
            if m.scenario_id not in by_scenario:
                by_scenario[m.scenario_id] = []
            by_scenario[m.scenario_id].append(m)
        
        aggregated = {}
        for scenario, metrics_list in by_scenario.items():
            aggregated[scenario] = self._compute_aggregates(metrics_list)
        
        return aggregated
    
    def _compute_aggregates(self, metrics_list: List[ExperimentMetrics]) -> Dict[str, Any]:
        """Compute aggregate statistics for a list of metrics."""
        if not metrics_list:
            return {}
        
        n = len(metrics_list)
        
        # Helper for safe stdev
        def safe_stdev(values):
            return stdev(values) if len(values) > 1 else 0.0
        
        token_reductions = [m.token_reduction_pct for m in metrics_list]
        turns = [m.total_turns for m in metrics_list]
        durations = [m.total_duration_seconds for m in metrics_list]
        
        return {
            "count": n,
            "token_reduction": {
                "mean": mean(token_reductions),
                "std": safe_stdev(token_reductions),
                "min": min(token_reductions),
                "max": max(token_reductions)
            },
            "turns": {
                "mean": mean(turns),
                "std": safe_stdev(turns),
                "min": min(turns),
                "max": max(turns)
            },
            "duration_seconds": {
                "mean": mean(durations),
                "std": safe_stdev(durations),
                "min": min(durations),
                "max": max(durations)
            },
            "total_tokens_used": sum(m.total_input_tokens for m in metrics_list),
            "total_tokens_if_full": sum(m.tokens_if_full_context for m in metrics_list)
        }
    
    def compare_strategies(self) -> Dict[str, Any]:
        """Compare strategies against the full_context baseline."""
        by_strategy = self.aggregate_by_strategy()
        
        if "full" not in by_strategy:
            return {"error": "No baseline (full_context) runs found"}
        
        baseline = by_strategy["full"]["token_reduction"]["mean"]
        
        comparison = {}
        for strategy, stats in by_strategy.items():
            if strategy == "full":
                comparison[strategy] = {
                    "is_baseline": True,
                    "token_reduction_mean": stats["token_reduction"]["mean"],
                    "improvement_over_baseline": 0.0
                }
            else:
                comparison[strategy] = {
                    "is_baseline": False,
                    "token_reduction_mean": stats["token_reduction"]["mean"],
                    "improvement_over_baseline": stats["token_reduction"]["mean"] - baseline
                }
        
        return comparison
    
    def to_dataframe_dict(self) -> List[Dict[str, Any]]:
        """Convert metrics to a list of dicts suitable for pandas DataFrame."""
        return [
            {
                "scenario_id": m.scenario_id,
                "persona_id": m.persona_id,
                "strategy": m.strategy,
                "run_id": m.run_id,
                "total_turns": m.total_turns,
                "total_goals": m.total_goals,
                "goals_achieved": m.goals_achieved,
                "token_reduction_pct": m.token_reduction_pct,
                "total_input_tokens": m.total_input_tokens,
                "tokens_if_full_context": m.tokens_if_full_context,
                "duration_seconds": m.total_duration_seconds,
                "gsr": m.gsr,
                "turn_success_rate": m.turn_success_rate
            }
            for m in self.metrics
        ]
    
    def save(self, filepath: str):
        """Save all metrics to a JSON file."""
        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_runs": len(self.metrics),
            "by_strategy": self.aggregate_by_strategy(),
            "by_scenario": self.aggregate_by_scenario(),
            "comparison": self.compare_strategies(),
            "raw_metrics": self.to_dataframe_dict()
        }
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct metrics from raw data
        for raw in data.get("raw_metrics", []):
            self.metrics.append(ExperimentMetrics(
                scenario_id=raw["scenario_id"],
                persona_id=raw["persona_id"],
                strategy=raw["strategy"],
                run_id=raw["run_id"],
                total_turns=raw["total_turns"],
                total_goals=raw["total_goals"],
                goals_achieved=raw["goals_achieved"],
                token_reduction_pct=raw["token_reduction_pct"],
                total_input_tokens=raw["total_input_tokens"],
                tokens_if_full_context=raw["tokens_if_full_context"],
                total_duration_seconds=raw["duration_seconds"],
                gsr=raw.get("gsr", 0.0),
                turn_success_rate=raw.get("turn_success_rate", 0.0)
            ))
    
    def generate_report(self) -> str:
        """Generate a human-readable report."""
        by_strategy = self.aggregate_by_strategy()
        comparison = self.compare_strategies()
        
        lines = [
            "=" * 60,
            "CONTEXT COMPACTION EXPERIMENT RESULTS",
            "=" * 60,
            f"\nTotal runs: {len(self.metrics)}",
            f"Strategies tested: {list(by_strategy.keys())}",
            "\n" + "-" * 40,
            "TOKEN REDUCTION BY STRATEGY",
            "-" * 40
        ]
        
        for strategy, stats in by_strategy.items():
            lines.append(
                f"\n{strategy.upper()}:\n"
                f"  Runs: {stats['count']}\n"
                f"  Token Reduction: {stats['token_reduction']['mean']:.1f}% "
                f"(±{stats['token_reduction']['std']:.1f}%)\n"
                f"  Avg Turns: {stats['turns']['mean']:.1f}\n"
                f"  Avg Duration: {stats['duration_seconds']['mean']:.1f}s"
            )
        
        lines.extend([
            "\n" + "-" * 40,
            "COMPARISON VS BASELINE (full_context)",
            "-" * 40
        ])
        
        for strategy, comp in comparison.items():
            if not comp.get("is_baseline"):
                lines.append(
                    f"\n{strategy}: "
                    f"{comp['improvement_over_baseline']:+.1f}% improvement"
                )
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
