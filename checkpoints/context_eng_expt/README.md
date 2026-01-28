# Context Engineering Experiment

Experimental framework for testing context management strategies in agentic AI systems, inspired by the "Mind The Goal" paper.

## Quick Start

### Run Fair Comparison (Recommended)
```bash
python context_eng_expt/run_replay_comparison.py
```

This runs the **Replay Mode** comparison which:
1. Generates a baseline conversation with simulated user
2. Replays the SAME user messages against all 3 strategies
3. Ensures 1:1 fair comparison (identical inputs, only context management differs)

### View Results
```bash
cat context_eng_expt/results/fair_replay_results.json | python -m json.tool
```

Or see the detailed analysis in `EXPERIMENT_RESULTS.md`.

---

## Strategies Compared

| Strategy | Description |
|----------|-------------|
| **Full Context** | Complete conversation history (baseline) |
| **Sliding Window** | Keep last N turns only |
| **Goal Boundary** | Summarize completed goals (from paper) |

## Metrics

| Metric | What It Measures |
|--------|------------------|
| **Token Reduction** | Input tokens saved vs full context |
| **Tool Accuracy** | % of successful tool calls |
| **GSR** | Goal Success Rate (evaluated by Judge Agent) |
| **Latency** | Total conversation time |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_replay_comparison.py` | **Recommended** - Fair 1:1 comparison |
| `evaluate_gsr_posthoc.py` | Evaluate existing conversations for GSR |
| `strategy_comparison.py` | Original comparison (different turn counts) |
| `smoke_test.py` | Quick infrastructure test |

---

## Architecture

```
context_eng_expt/
├── agents/
│   ├── chatbot_agent.py    # Main chatbot with context strategy
│   ├── simulated_user.py   # LLM-based user simulator
│   └── goal_detector.py    # Detects goal boundaries
├── context/
│   ├── strategies.py       # Full, Sliding, GoalBoundary strategies
│   └── goal_summarizer.py  # Summarizes completed goals
├── tools/
│   ├── router.py           # Routes LLM tool calls
│   ├── search.py           # Web search tool
│   ├── scraper.py          # Web scraper tool
│   └── code_executor.py    # Python execution tool
├── results/                # JSON experiment results
└── EXPERIMENT_RESULTS.md   # Detailed findings
```

---

## Key Findings

From fair replay comparison (same user messages):

| Strategy | Token Reduction | Tool Accuracy | GSR |
|----------|-----------------|---------------|-----|
| Full Context | 0.0% | 50% | 50% |
| Sliding Window | **70.0%** | 50% | 50% |
| Goal Boundary | 47.7% | **60%** | 50% |

**Conclusion**: Context compression strategies save tokens without hurting task performance. Goal boundary achieves best tool accuracy.

---

## Configuration

Edit `config.py` to change:
- Goals and personas
- Max turns
- Window size for sliding window
- Tool settings

---

## Known Issues

1. `execute_python` tool has argument parsing bug (first call often fails)
2. Weather lookup fails due to lack of real-time data API
3. Single run - need multiple iterations for statistical significance
