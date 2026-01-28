# Context Compaction Experiment Design

## Overview

This experiment explores **goal-boundary-aware context compaction** for conversational AI systems. The core hypothesis is that using semantic goal boundaries (detected via the MindTheGoal framework) as natural compression points is more effective than arbitrary token-based truncation.

---

## Problem Statement

### The Context Window Challenge
Large Language Models have finite context windows. As conversations grow:
- Token costs increase linearly
- Latency increases with input size
- Eventually, older context must be truncated or summarized

### Current Approaches (Limitations)
| Approach | How it Works | Limitation |
|----------|--------------|------------|
| **FIFO Truncation** | Drop oldest messages | Loses important early context |
| **Sliding Window** | Keep last N turns | Arbitrary cutoff, may split goals |
| **Fixed Summarization** | Summarize every N turns | Breaks mid-goal, loses nuance |
| **Token Budget** | Compress when >K tokens | No semantic awareness |

### Our Hypothesis
**Goal boundaries are natural semantic compression points.** When a user starts a new goal, the detailed turn-by-turn history of the previous goal can be safely compressed into a summary without losing task-relevant information.

---

## The Idea: Goal-Boundary Compaction

### Core Concept
```
┌─────────────────────────────────────────────────────────────────┐
│ Traditional Approach: Arbitrary Truncation                      │
├─────────────────────────────────────────────────────────────────┤
│ [Turn1][Turn2][Turn3][Turn4] | ← truncate here | [Turn5][Turn6] │
│        ↑ might be mid-goal!                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Our Approach: Goal-Boundary Compression                         │
├─────────────────────────────────────────────────────────────────┤
│ [Goal1: "User booked Italian restaurant"] ← summary            │
│ [Turn5][Turn6] ← full detail for current goal                  │
│        ↑ compress at goal boundary                              │
└─────────────────────────────────────────────────────────────────┘
```

### When to Compress
The MindTheGoal framework detects `is_new_goal=True` for turns that start a new user intent. This is the trigger for compression.

### What Gets Compressed
- All turns from the **completed goal** → single summary
- Summary captures: what the user wanted, what was achieved, key decisions

### Example Flow
```
Turn 1: "I need a restaurant in the center"
Turn 2: "Italian please, under £30"
Turn 3: "Yes, book 'Pasta Palace' for 2 at 7pm"
Turn 4: "Thanks! Now I need a taxi to get there"  ← is_new_goal=True

At Turn 4:
- Compress Turns 1-3 into: "User booked 'Pasta Palace' (Italian, center, £30, 2 people, 7pm)"
- Context becomes: [Summary] + Turn 4
```

---

## Architecture

### Components
```
context_eng_expt/
├── agents/
│   ├── simulated_user.py       # LLM-based user playing a role with goals
│   ├── chatbot_agent.py        # The conversational agent being tested
│   └── goal_detector.py        # Wrapper around MindTheGoal's detector
├── context/
│   ├── context_manager.py      # Core context tracking & compression logic
│   ├── goal_summarizer.py      # LLM-based summarization of completed goals
│   └── strategies.py           # Different compaction strategies
├── experiment/
│   ├── runner.py               # Orchestrates the experiment loop
│   ├── scenarios.py            # Defines user personas and goal sets
│   └── metrics.py              # Collects and computes metrics
├── results/                    # Output directory for experiment data
├── config.py                   # Experiment configuration
└── README.md                   # Usage documentation
```

### Data Flow
```
                                    ┌─────────────────┐
                                    │   Experiment    │
                                    │   Scenarios     │
                                    │ (goals/personas)│
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────┐                ┌─────────────────┐
│  Simulated User │◄───────────────│  Conversation   │
│  (LLM + Goals)  │                │  Orchestrator   │
└────────┬────────┘                └────────┬────────┘
         │                                  │
         │ user_message                     │
         ▼                                  │
┌─────────────────┐                         │
│  Goal Detector  │                         │
│ (is_new_goal?)  │                         │
└────────┬────────┘                         │
         │                                  │
         │ is_new_goal                      │
         ▼                                  │
┌─────────────────┐                         │
│ Context Manager │◄────────────────────────┘
│ - Track turns   │        config (strategy)
│ - Compress if   │
│   new goal      │
│ - Build context │
└────────┬────────┘
         │
         │ context (full or compressed)
         ▼
┌─────────────────┐
│  Chatbot Agent  │
│  (LLM response) │
└────────┬────────┘
         │
         │ agent_response
         ▼
┌─────────────────┐
│    Metrics      │
│  Collector      │
└─────────────────┘
```

---

## User Simulation

### Why Simulate?
We need controlled, reproducible conversations to compare strategies. Real user studies are expensive and variable.

### Approaches Considered

#### Option 1: LLM-as-Simulated-User (Recommended)
**Pros:**
- Generates realistic, varied utterances
- Can simulate different user personalities
- Adapts to chatbot responses naturally
- Can be given explicit goals to pursue

**Cons:**
- Additional LLM cost
- Needs careful prompting to stay goal-focused
- May "give up" too easily or be too cooperative

**Implementation:**
```python
class SimulatedUser:
    def __init__(self, goals: List[str], persona: str):
        self.goals = goals
        self.current_goal_idx = 0
        self.persona = persona
        self.llm = BedrockClient()
    
    async def generate_utterance(self, conversation_history: List[Turn]) -> str:
        prompt = f"""
        You are playing a user with the following characteristics:
        Persona: {self.persona}
        
        Your goals (in order):
        {self._format_goals()}
        
        Current goal: {self.goals[self.current_goal_idx]}
        
        Conversation so far:
        {self._format_history(conversation_history)}
        
        Generate the next user message. Be natural, may ask clarifying questions,
        express preferences, or move to next goal if current one is satisfied.
        """
        return await self.llm.invoke(prompt)
```

#### Option 2: Replay from MultiWOZ
**Pros:**
- Uses real human utterances
- Zero additional LLM cost for user side
- Perfectly reproducible

**Cons:**
- Chatbot responses may differ from original, making user's next turn awkward
- Limited to existing dialogues
- No adaptation to different chatbot behaviors

**Implementation:**
```python
class ReplayUser:
    def __init__(self, dialogue: List[Turn]):
        self.dialogue = dialogue
        self.turn_idx = 0
    
    def get_next_utterance(self) -> str:
        utterance = self.dialogue[self.turn_idx].user_message
        self.turn_idx += 1
        return utterance
```

#### Option 3: Hybrid (Goal-Seeded Generation)
**Pros:**
- Real goals from MultiWOZ
- Fresh, adaptive utterances
- Balance of control and flexibility

**Cons:**
- Moderate complexity
- Still has LLM cost

**Implementation:**
```python
class HybridUser:
    def __init__(self, goals_from_multiwoz: List[str]):
        self.goals = goals_from_multiwoz
        self.llm = BedrockClient()
    
    async def generate_utterance(self, history, current_goal):
        # Generate new utterance based on goal, not script
        ...
```

### Recommendation
**Start with Option 1 (LLM-as-User)** for flexibility and realism. Add Option 2 as a baseline comparison.

---

## Context Compaction Strategies

### Strategy 1: Full Context (Baseline)
Keep all turns in context, no compression.
```python
class FullContextStrategy:
    def get_context(self, turns: List[Turn]) -> str:
        return "\n".join([t.format() for t in turns])
```

### Strategy 2: Sliding Window
Keep last N turns regardless of goal boundaries.
```python
class SlidingWindowStrategy:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def get_context(self, turns: List[Turn]) -> str:
        recent = turns[-self.window_size:]
        return "\n".join([t.format() for t in recent])
```

### Strategy 3: Goal-Boundary Compaction (Our Approach)
Compress completed goals into summaries.
```python
class GoalBoundaryStrategy:
    def __init__(self, summarizer: GoalSummarizer):
        self.summarizer = summarizer
        self.goal_summaries: List[str] = []
        self.current_goal_turns: List[Turn] = []
    
    async def on_new_turn(self, turn: Turn, is_new_goal: bool):
        if is_new_goal and self.current_goal_turns:
            # Compress previous goal
            summary = await self.summarizer.summarize(self.current_goal_turns)
            self.goal_summaries.append(summary)
            self.current_goal_turns = []
        self.current_goal_turns.append(turn)
    
    def get_context(self) -> str:
        summaries = "\n".join([f"[Previous: {s}]" for s in self.goal_summaries])
        current = "\n".join([t.format() for t in self.current_goal_turns])
        return f"{summaries}\n{current}"
```

### Strategy 4: Token Budget with Goal Awareness
Compress oldest goals first when budget exceeded.
```python
class TokenBudgetStrategy:
    def __init__(self, max_tokens: int = 4000, summarizer: GoalSummarizer):
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.goals: List[Goal] = []  # Each goal has turns or summary
    
    async def get_context(self) -> str:
        context = self._build_context()
        while self._count_tokens(context) > self.max_tokens:
            # Compress oldest non-summarized goal
            for goal in self.goals:
                if not goal.is_summarized:
                    goal.summary = await self.summarizer.summarize(goal.turns)
                    goal.is_summarized = True
                    break
            context = self._build_context()
        return context
```

---

## Goal Summarization

### Summarizer Prompt
```python
SUMMARIZATION_PROMPT = """
Summarize this completed conversation goal in 1-2 sentences.
Focus on: what the user wanted, what was achieved, key details.

Goal turns:
{turns}

Summary (be concise):
"""
```

### Summary Format Options
1. **Narrative**: "User booked a table at Pasta Palace for 2 at 7pm"
2. **Structured**: "BOOKING: restaurant=Pasta Palace, guests=2, time=7pm, status=confirmed"
3. **Hybrid**: "✓ Restaurant booking: Pasta Palace, 2 guests, 7pm"

### Quality vs Token Trade-off
- More detailed summary = better context but more tokens
- Too brief = may lose important nuances
- Experiment with different summary lengths

---

## Metrics

### Primary Metrics

#### 1. Token Reduction
```python
token_reduction = (full_context_tokens - compressed_tokens) / full_context_tokens * 100
```

#### 2. GSR Impact
Compare Goal Success Rate across strategies:
```python
gsr_delta = strategy_gsr - baseline_gsr
```
Negative delta = compaction hurt quality

#### 3. Response Quality
Judge responses with same LLM-as-judge from MindTheGoal:
- Did the chatbot understand the user?
- Was the response helpful?

### Secondary Metrics

#### 4. Compression Ratio
```python
compression_ratio = summary_tokens / original_turns_tokens
```

#### 5. Latency
Time from user message to agent response:
```python
latency_improvement = (baseline_latency - strategy_latency) / baseline_latency * 100
```

#### 6. Information Retention
Manual evaluation: Did important details survive compression?

### Metrics Collection
```python
@dataclass
class ExperimentMetrics:
    strategy: str
    scenario_id: str
    total_turns: int
    total_goals: int
    
    # Token metrics
    total_input_tokens: int
    tokens_if_full_context: int
    token_reduction_pct: float
    
    # Quality metrics
    gsr: float
    turn_success_rate: float
    
    # Efficiency metrics
    avg_latency_ms: float
    total_cost_usd: float
    
    # Per-goal breakdown
    goal_summaries: List[str]
    compression_ratios: List[float]
```

---

## Experiment Design

### Experimental Variables
- **Independent**: Compaction strategy (4 strategies)
- **Dependent**: Token usage, GSR, latency, cost
- **Controlled**: Same scenarios, same LLM, same prompts

### Scenarios
```python
scenarios = [
    {
        "id": "single_goal_short",
        "description": "Single restaurant booking",
        "goals": ["Find and book an Italian restaurant"],
        "expected_turns": 3-5
    },
    {
        "id": "single_goal_long",
        "description": "Complex booking with many preferences",
        "goals": ["Book restaurant: Italian, center, cheap, vegetarian options, 4 people, Friday 7pm"],
        "expected_turns": 6-10
    },
    {
        "id": "multi_goal_sequential",
        "description": "Restaurant then taxi",
        "goals": [
            "Book Italian restaurant in center",
            "Book taxi to the restaurant"
        ],
        "expected_turns": 8-12
    },
    {
        "id": "multi_goal_complex",
        "description": "Hotel, restaurant, and taxi",
        "goals": [
            "Find hotel near station for 2 nights",
            "Book restaurant near hotel",
            "Arrange taxi from station to hotel"
        ],
        "expected_turns": 15-20
    }
]
```

### User Personas
```python
personas = [
    {
        "id": "efficient",
        "description": "Knows what they want, concise, doesn't chat",
        "prompt_modifier": "Be direct and efficient. State requirements clearly."
    },
    {
        "id": "exploratory", 
        "description": "Asks questions, changes mind, explores options",
        "prompt_modifier": "Ask questions, consider alternatives, change your mind once."
    },
    {
        "id": "confused",
        "description": "Unclear requirements, needs guidance",
        "prompt_modifier": "Be vague initially. Let the assistant guide you."
    }
]
```

### Experiment Matrix
| Scenario | Persona | Strategy | Runs |
|----------|---------|----------|------|
| single_short | efficient | full | 3 |
| single_short | efficient | sliding | 3 |
| single_short | efficient | goal_boundary | 3 |
| ... | ... | ... | ... |

Total: 4 scenarios × 3 personas × 4 strategies × 3 runs = 144 conversations

---

## Expected Results

### Hypothesis 1: Token Reduction
**Expectation**: Goal-boundary compaction reduces tokens by 40-60% for multi-goal conversations while maintaining quality.

### Hypothesis 2: GSR Preservation
**Expectation**: GSR should be within ±5% of baseline (full context). If significantly lower, summaries are losing critical information.

### Hypothesis 3: Latency Improvement
**Expectation**: 20-30% latency reduction for longer conversations due to smaller context.

### Hypothesis 4: Superiority over Sliding Window
**Expectation**: Goal-boundary beats sliding window because it doesn't split goals arbitrarily.

---

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create folder structure
2. Implement `SimulatedUser` with LLM
3. Implement `ChatbotAgent` with pluggable context strategy
4. Implement `GoalDetector` wrapper

### Phase 2: Compaction Strategies
1. Implement `FullContextStrategy`
2. Implement `SlidingWindowStrategy`
3. Implement `GoalBoundaryStrategy`
4. Implement `GoalSummarizer`

### Phase 3: Experiment Framework
1. Define scenarios and personas
2. Implement `ExperimentRunner`
3. Implement metrics collection
4. Create results storage

### Phase 4: Run and Analyze
1. Run experiment matrix
2. Aggregate results
3. Statistical analysis
4. Generate report

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Goal detection errors | Wrong compression timing | Validate on MultiWOZ first |
| Summarization loses info | GSR drops | Tune prompts, try structured summaries |
| Simulated user unrealistic | Invalid results | Compare with MultiWOZ replay |
| LLM cost too high | Can't run full matrix | Reduce runs, use smaller scenarios |
| Results not significant | No clear winner | More runs, longer scenarios |

---

## Tool System

### Why Real Tools?

To measure accuracy realistically, the chatbot needs to use actual tools. Without tools, we can only measure conversational quality. With tools, we can measure:

1. **Tool Selection**: Does the agent pick the right tool?
2. **Tool Arguments**: Are search queries, URLs, etc. correct?
3. **Result Integration**: Does the agent use tool results properly?
4. **Compression Impact**: Does summarization affect tool use accuracy?

### Tool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ChatbotAgent                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                Context Manager                        │  │
│  │  [Goal Summaries] + [Current Goal Turns]              │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    LLM (Claude)                       │  │
│  │  "I need to search for Italian restaurants..."        │  │
│  │  → Tool call: web_search("Italian restaurants NYC")   │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Tool Router                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │ web_search  │  │scrape_page  │  │execute_python│   │  │
│  │  │  (DDG)      │  │   (BS4)     │  │ (sandbox)    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Available Tools

| Tool | Implementation | Safety |
|------|----------------|--------|
| `web_search` | DuckDuckGo API via `duckduckgo-search` | Rate limited |
| `scrape_webpage` | `requests` + `BeautifulSoup` | Timeout, URL validation, max length |
| `execute_python` | Sandboxed `exec()` | Limited builtins, no file/network, timeout |

### Tool Call Format

The LLM outputs tool calls in XML format:
```xml
<tool_use>
<name>web_search</name>
<arguments>{"query": "Italian restaurants NYC", "max_results": 5}</arguments>
</tool_use>
```

The `ToolRouter` parses these and executes the appropriate tool.

### Tool Use Loop

```python
async def _respond_with_tools(self, context: str) -> str:
    for iteration in range(self.max_tool_iterations):
        response = await self._llm.invoke(context)
        tool_calls = self.tool_router.parse_tool_calls(response)
        
        if not tool_calls:
            return response  # Final answer
        
        # Execute tools and add results to context
        for tool_call in tool_calls:
            result = await self.tool_router.execute(tool_call)
            context += f"\nTool Result: {result}"
```

---

## Accuracy vs Token Trade-off Analysis

### The Core Question

**How much accuracy do we sacrifice for token savings?**

This is the key metric for evaluating context compaction strategies.

### Measuring Accuracy

#### 1. Task Completion (GSR)
```python
task_accuracy = goals_completed / total_goals
```

#### 2. Tool Accuracy
```python
tool_accuracy = successful_tool_calls / total_tool_calls
```

#### 3. Response Relevance (LLM Judge)
```python
relevance_score = judge_llm.evaluate(response, context, user_intent)  # 0-1
```

### Trade-off Visualization

```
Accuracy
    │
100%├───●─────────────────  ← Full Context (baseline)
    │    ╲
 95%├─────●───────────────  ← Goal Boundary
    │      ╲
 90%├───────●─────────────  ← Token Budget
    │        ╲
 85%├─────────╲───────────  
    │          ╲
 80%├───────────●─────────  ← Sliding Window
    │
    └─────┬─────┬─────┬─────┬──→ Token Reduction %
          10%   25%   40%   55%
```

### Computing Trade-off Efficiency

```python
@dataclass
class TradeoffMetrics:
    strategy: str
    token_reduction_pct: float    # How much tokens saved
    accuracy_drop_pct: float      # How much accuracy lost
    efficiency: float             # token_reduction / (1 + accuracy_drop)
```

**Efficiency Score**:
```python
efficiency = token_reduction_pct / (1 + accuracy_drop_pct)
```

Higher efficiency = better trade-off (more savings per accuracy point lost)

### Expected Trade-off Table

| Strategy | Token Savings | Accuracy | Efficiency |
|----------|--------------|----------|------------|
| Full Context | 0% | 100% | 0.0 |
| Sliding Window | 45% | 78% | 2.0 |
| Token Budget | 40% | 88% | 3.3 |
| **Goal Boundary** | **50%** | **94%** | **8.3** |

### Why Goal Boundary Should Win

1. **Semantic preservation**: Summaries capture goal outcomes, not arbitrary text
2. **Complete goals**: Never splits a goal mid-conversation
3. **Relevant context**: Current goal gets full detail
4. **Diminishing returns**: Old goal details rarely needed verbatim

---

## Future Extensions

### Phase 2: Personalization
- User preference profiles
- Communication style adaptation
- Historical context integration

### Phase 3: Learning & Memory
- Cross-session memory (vector DB)
- Preference learning from interactions
- Long-term storage (SQLite)
- Retrieval-augmented context

### Other Extensions
1. **Adaptive Compression**: Adjust summary detail based on goal importance
2. **Cross-Goal References**: Handle "same restaurant as before" type references
3. **Incremental Summarization**: Update summary as goal progresses
4. **Multi-Modal**: Support image/document context compaction
5. **Production Integration**: Deploy in real chatbot system

---

## References

- MindTheGoal Paper: [arXiv:2510.03696](https://arxiv.org/abs/2510.03696)
- MultiWOZ Dataset: [multi-woz.github.io](https://multi-woz.github.io/)
- Context Window Research: [Anthropic Context Windows](https://www.anthropic.com/news/100k-context-windows)
