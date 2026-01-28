# MindTheGoal - System Design Document

**Version**: 1.0  
**Date**: January 2026  
**Based on**: arXiv:2510.03696 - "Mind the Goal: Data-Efficient Goal-Oriented Evaluation of Conversational Agents and Chatbots using Teacher Models"

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Overview](#3-system-overview)
4. [Core Framework Design](#4-core-framework-design)
5. [Data Models](#5-data-models)
6. [LLM Integration](#6-llm-integration)
7. [API Design](#7-api-design)
8. [Frontend Architecture](#8-frontend-architecture)
9. [Dataset Management](#9-dataset-management)
10. [Evaluation Pipeline](#10-evaluation-pipeline)
11. [Error Handling & Resilience](#11-error-handling--resilience)
12. [Security Considerations](#12-security-considerations)
13. [Performance Optimization](#13-performance-optimization)
14. [Future Enhancements](#14-future-enhancements)

---

## 1. Executive Summary

MindTheGoal is a comprehensive implementation of a goal-oriented evaluation framework for multi-turn conversational agents. Unlike traditional turn-level metrics (BLEU, ROUGE, BERTScore), this system evaluates conversations at the **goal level**, measuring whether users actually achieve their intended objectives.

### Key Differentiators

| Traditional Evaluation | MindTheGoal |
|------------------------|-------------|
| Turn-level metrics | Goal-level metrics |
| No failure diagnosis | RCOF taxonomy (7 error types) |
| Requires reference texts | Uses LLM-as-judge |
| Offline only | Real-time + batch evaluation |

---

## 2. Problem Statement

### 2.1 Challenges with Turn-Level Evaluation

1. **Misaligned Success Signals**: A conversation might have good turn-level scores but fail to achieve the user's goal
2. **No Actionable Insights**: Turn metrics don't explain WHY conversations fail
3. **Multi-turn Blindness**: Existing metrics don't capture the compounding effect of errors across turns

### 2.2 Goals of This System

1. **Accurate Goal Measurement**: Compute Goal Success Rate (GSR) with strict criteria
2. **Actionable Diagnostics**: Classify failures using RCOF taxonomy
3. **Real-time Feedback**: Provide live evaluation during interactive chat
4. **Scalable Batch Processing**: Evaluate large datasets efficiently

---

## 3. System Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    React Frontend                           │    │
│  │  ┌─────────────┐ ┌──────────────┐ ┌───────────────────┐    │    │
│  │  │    Chat     │ │ Judge Panel  │ │   GSR Dashboard   │    │    │
│  │  │  Interface  │ │   (Live)     │ │  (Visualizations) │    │    │
│  │  └──────┬──────┘ └──────┬───────┘ └─────────┬─────────┘    │    │
│  └─────────┼───────────────┼───────────────────┼──────────────┘    │
└────────────┼───────────────┼───────────────────┼────────────────────┘
             │ WebSocket     │ REST              │ REST
┌────────────▼───────────────▼───────────────────▼────────────────────┐
│                           API LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    FastAPI Backend                          │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │    │
│  │  │  /api/chat   │ │ /api/eval    │ │  /api/datasets   │    │    │
│  │  │  (WebSocket) │ │ (REST)       │ │  (REST)          │    │    │
│  │  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘    │    │
│  └─────────┼───────────────┼───────────────────┼──────────────┘    │
└────────────┼───────────────┼───────────────────┼────────────────────┘
             │               │                   │
┌────────────▼───────────────▼───────────────────▼────────────────────┐
│                         SERVICE LAYER                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐    │    │
│  │  │   Chat Service   │  │     Evaluation Service       │    │    │
│  │  │ • Session Mgmt   │  │ • Batch Processing           │    │    │
│  │  │ • Message Flow   │  │ • Report Generation          │    │    │
│  │  └────────┬─────────┘  └──────────────┬───────────────┘    │    │
│  └───────────┼───────────────────────────┼────────────────────┘    │
└──────────────┼───────────────────────────┼──────────────────────────┘
               │                           │
┌──────────────▼───────────────────────────▼──────────────────────────┐
│                         CORE FRAMEWORK                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  ┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐ │     │
│  │  │ Goal Segmentation│ │GSR Calculator│ │RCOF Classifier │ │     │
│  │  │                  │ │             │ │                 │ │     │
│  │  │ • Boundary Det.  │ │ • Strict    │ │ • E1-E7 Codes   │ │     │
│  │  │ • is_new_goal()  │ │ • Aggregate │ │ • Attribution   │ │     │
│  │  └──────────────────┘ └─────────────┘ └─────────────────┘ │     │
│  └────────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                  Turn Evaluator                            │     │
│  │  • Success/Failure determination                           │     │
│  │  • Quality signals integration                             │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
               │                           │
┌──────────────▼───────────────────────────▼──────────────────────────┐
│                          LLM LAYER                                  │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                  AWS Bedrock Client                        │     │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐ │     │
│  │  │     Chat Agent      │  │       Judge Agent           │ │     │
│  │  │                     │  │                             │ │     │
│  │  │ • Claude 3.7 Sonnet │  │ • Claude 3.7 Sonnet + CoT   │ │     │
│  │  │ • Temp: 0.7         │  │ • Temp: 0.1                 │ │     │
│  │  │ • Conversational    │  │ • <think> reasoning         │ │     │
│  │  └─────────────────────┘  └─────────────────────────────┘ │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Frontend** | User interaction, real-time display, visualizations |
| **API Layer** | Request routing, WebSocket management, authentication |
| **Service Layer** | Business logic, session management, orchestration |
| **Core Framework** | GSR/RCOF computation, goal segmentation |
| **LLM Layer** | Chat generation, evaluation judgments |

---

## 4. Core Framework Design

### 4.1 Conversation Hierarchy

```
Session (S)
├── Goal 1 (G₁)
│   ├── Turn 1 (T₁): (query₁, response₁)
│   └── Turn 2 (T₂): (query₂, response₂)
├── Goal 2 (G₂)
│   ├── Turn 3 (T₃): (query₃, response₃)
│   ├── Turn 4 (T₄): (query₄, response₄)
│   └── Turn 5 (T₅): (query₅, response₅)
└── Goal 3 (G₃)
    └── Turn 6 (T₆): (query₆, response₆)
```

### 4.2 Goal Segmentation Algorithm

```python
def segment_goals(turns: List[Turn]) -> List[Goal]:
    """
    Segment conversation turns into distinct goals.
    
    Algorithm:
    1. First turn always starts a new goal
    2. For each subsequent turn, ask LLM: "Is this a new goal?"
    3. Group contiguous turns into goals
    """
    goals = []
    current_goal_turns = []
    
    for i, turn in enumerate(turns):
        if i == 0:
            is_new_goal = True
        else:
            is_new_goal = llm_judge.is_new_goal(
                current_turn=turn,
                previous_turns=current_goal_turns,
                conversation_context=turns[:i]
            )
        
        if is_new_goal and current_goal_turns:
            goals.append(Goal(turns=current_goal_turns))
            current_goal_turns = []
        
        current_goal_turns.append(turn)
    
    # Add final goal
    if current_goal_turns:
        goals.append(Goal(turns=current_goal_turns))
    
    return goals
```

### 4.3 GSR Calculation

```python
def calculate_gsr(goals: List[Goal]) -> GSRResult:
    """
    Calculate Goal Success Rate.
    
    A goal is successful IFF all its turns are successful.
    GSR = (successful_goals / total_goals) × 100
    """
    successful_goals = 0
    failed_goals = []
    
    for goal in goals:
        all_turns_successful = all(
            turn.quality == TurnQuality.SUCCESS 
            for turn in goal.turns
        )
        
        if all_turns_successful:
            successful_goals += 1
        else:
            # Find first failed turn for RCOF attribution
            first_failure = next(
                t for t in goal.turns 
                if t.quality == TurnQuality.FAILURE
            )
            failed_goals.append(FailedGoal(
                goal=goal,
                rcof=first_failure.rcof,
                first_failed_turn=first_failure
            ))
    
    gsr = (successful_goals / len(goals)) * 100
    
    return GSRResult(
        total_goals=len(goals),
        successful_goals=successful_goals,
        gsr_percentage=gsr,
        failed_goals=failed_goals
    )
```

### 4.4 RCOF Taxonomy Implementation

```python
class RCOF(Enum):
    """Root Cause of Failure codes from the paper."""
    
    E1_LANGUAGE_UNDERSTANDING = "E1"  # Misunderstood request/context
    E2_REFUSAL_TO_ANSWER = "E2"       # Inappropriate refusal
    E3_INCORRECT_RETRIEVAL = "E3"     # Wrong documents retrieved
    E4_RETRIEVAL_FAILURE = "E4"       # No documents retrieved
    E5_SYSTEM_ERROR = "E5"            # Timeout, truncation, etc.
    E6_INCORRECT_ROUTING = "E6"       # Wrong domain/module
    E7_OUT_OF_DOMAIN = "E7"           # Outside system scope
    
    @classmethod
    def get_description(cls, code: str) -> str:
        descriptions = {
            "E1": "Language Understanding Failure - Misunderstood user's request",
            "E2": "Refusal to Answer - Inappropriate refusal despite ability",
            "E3": "Incorrect Retrieval - Retrieved wrong information",
            "E4": "Retrieval Failure - Failed to retrieve any information",
            "E5": "System Error - Technical issues (timeout, truncation)",
            "E6": "Incorrect Routing - Query routed to wrong domain",
            "E7": "Out-of-Domain - Request outside system scope"
        }
        return descriptions.get(code, "Unknown error code")
```

---

## 5. Data Models

### 5.1 Core Domain Models

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from datetime import datetime
import uuid

class TurnQuality(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"  # Not yet evaluated

@dataclass
class Turn:
    """A single turn in a conversation (query + response)."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int = 0
    user_message: str = ""
    agent_response: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Evaluation fields
    quality: TurnQuality = TurnQuality.PENDING
    rcof: Optional[str] = None  # E1-E7 if failure
    is_new_goal: bool = False
    
    # Judge output
    judge_reasoning: Optional[str] = None  # <think> content

@dataclass
class Goal:
    """A user goal comprising one or more turns."""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_number: int = 0
    turns: List[Turn] = field(default_factory=list)
    
    # Computed fields
    quality: TurnQuality = TurnQuality.PENDING
    rcof: Optional[str] = None  # From first failed turn
    
    @property
    def is_successful(self) -> bool:
        return all(t.quality == TurnQuality.SUCCESS for t in self.turns)
    
    @property
    def turn_count(self) -> int:
        return len(self.turns)

@dataclass
class Session:
    """A complete conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goals: List[Goal] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    @property
    def total_turns(self) -> int:
        return sum(g.turn_count for g in self.goals)
    
    @property
    def gsr(self) -> float:
        if not self.goals:
            return 0.0
        successful = sum(1 for g in self.goals if g.is_successful)
        return (successful / len(self.goals)) * 100

@dataclass
class EvaluationResult:
    """Complete evaluation result for a dataset."""
    evaluation_id: str
    dataset_name: str
    total_sessions: int
    total_goals: int
    total_turns: int
    
    overall_gsr: float
    single_turn_gsr: float
    multi_turn_gsr: float
    
    rcof_distribution: dict  # {RCOF code: count}
    domain_gsr: dict  # {domain: gsr} if applicable
    
    evaluated_at: datetime
    sessions: List[Session]  # Detailed results
```

### 5.2 API Request/Response Models

```python
from pydantic import BaseModel
from typing import Optional, List

# Chat API
class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    turn_number: int
    agent_response: str
    
    # Live evaluation
    is_new_goal: bool
    goal_number: int
    turn_quality: str  # success/failure/pending
    rcof: Optional[str] = None
    judge_reasoning: Optional[str] = None
    current_gsr: float

# Evaluation API
class EvaluationRequest(BaseModel):
    dataset: str  # "multiwoz" or "custom"
    sample_size: Optional[int] = None
    custom_path: Optional[str] = None

class EvaluationStatus(BaseModel):
    job_id: str
    status: str  # pending/running/completed/failed
    progress: float  # 0-100
    message: Optional[str] = None

class GSRReport(BaseModel):
    evaluation_id: str
    dataset_name: str
    overall_gsr: float
    single_turn_gsr: float
    multi_turn_gsr: float
    total_goals: int
    total_sessions: int
    rcof_distribution: dict
    evaluated_at: str
```

---

## 6. LLM Integration

### 6.1 AWS Bedrock Client Architecture

```python
class BedrockClient:
    """
    Wrapper for AWS Bedrock Claude integration.
    
    Features:
    - Automatic credential refresh on expiration
    - Retry logic with exponential backoff
    - Rate limiting support
    - Structured output parsing
    """
    
    def __init__(self, config: Settings):
        self.config = config
        self.model_id = config.bedrock_model_id
        self.region = config.aws_region
        self._init_client()
    
    def _init_client(self):
        """Initialize Bedrock client with LangChain."""
        from langchain_aws import ChatBedrock
        
        self.llm = ChatBedrock(
            model_id=self.model_id,
            region_name=self.region,
            model_kwargs={
                "temperature": self.config.default_temperature,
                "max_tokens": self.config.max_tokens
            }
        )
    
    async def invoke(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Invoke the model with retry logic."""
        # Implementation with retry on expired tokens
        pass
```

### 6.2 Judge Agent Design

```python
class JudgeAgent:
    """
    LLM-as-judge for evaluating conversation quality.
    
    Uses Chain-of-Thought prompting with <think> tags
    for interpretable reasoning.
    """
    
    SYSTEM_PROMPT = """
You are an expert evaluator assessing chatbot conversation quality.

For each turn, you must determine:
1. is_new_goal: Does this turn start a new user goal? (yes/no)
2. quality: Was the bot response successful? (success/failure)
3. rcof: If failure, what is the root cause? (E1-E7)

RCOF Codes:
- E1: Language Understanding Failure - Bot misunderstood the user
- E2: Refusal to Answer - Bot refused when it should have answered
- E3: Incorrect Retrieval - Bot retrieved wrong information
- E4: Retrieval Failure - Bot failed to retrieve any information
- E5: System Error - Technical issues (timeout, truncation)
- E6: Incorrect Routing - Query went to wrong domain/module
- E7: Out-of-Domain - Request outside system's capabilities

Instructions:
1. Analyze the conversation context carefully
2. Reason step-by-step inside <think>...</think> tags
3. Output your judgment as JSON after reasoning

Output Format:
<think>
[Your detailed reasoning about the turn quality]
</think>
{
    "turn_number": N,
    "is_new_goal": "yes" or "no",
    "quality": "success" or "failure",
    "rcof": "E1" to "E7" if failure, null if success
}
"""
    
    async def evaluate_turn(
        self,
        current_turn: Turn,
        conversation_history: List[Turn],
        goal_context: Optional[str] = None
    ) -> TurnEvaluation:
        """Evaluate a single turn in context."""
        
        prompt = self._build_evaluation_prompt(
            current_turn, 
            conversation_history,
            goal_context
        )
        
        response = await self.bedrock_client.invoke(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for consistency
        )
        
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> TurnEvaluation:
        """Extract reasoning and JSON from response."""
        # Extract <think>...</think> content
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        # Extract JSON
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return TurnEvaluation(
                is_new_goal=data.get("is_new_goal") == "yes",
                quality=TurnQuality(data.get("quality", "pending")),
                rcof=data.get("rcof"),
                reasoning=reasoning
            )
        raise ValueError("Failed to parse judge response")
```

### 6.3 Chat Agent Design

```python
class ChatAgent:
    """
    Conversational agent for interactive chat.
    
    Can be configured with different personas/capabilities.
    """
    
    DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant that helps users with various tasks.
Be clear, concise, and accurate in your responses.
If you don't know something, say so rather than making up information.
"""
    
    def __init__(
        self,
        bedrock_client: BedrockClient,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ):
        self.bedrock_client = bedrock_client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
    
    async def respond(
        self,
        user_message: str,
        conversation_history: List[Turn]
    ) -> str:
        """Generate a response to the user's message."""
        
        messages = self._build_messages(user_message, conversation_history)
        
        response = await self.bedrock_client.invoke(
            messages=messages,
            temperature=self.temperature
        )
        
        return response
```

---

## 7. API Design

### 7.1 REST Endpoints

```yaml
# Evaluation Endpoints
POST /api/evaluate
  Description: Start a batch evaluation job
  Request:
    dataset: string (required) - "multiwoz" or "custom"
    sample_size: int (optional) - Number of dialogues
    custom_path: string (optional) - Path to custom dataset
  Response:
    job_id: string
    status: "pending"

GET /api/evaluate/{job_id}
  Description: Get evaluation job status
  Response:
    job_id: string
    status: string
    progress: float
    result: GSRReport (if completed)

GET /api/evaluate/{job_id}/report
  Description: Get detailed evaluation report
  Response:
    full GSRReport with session details

# Dataset Endpoints
GET /api/datasets
  Description: List available datasets
  Response:
    datasets: List[DatasetInfo]

POST /api/datasets/upload
  Description: Upload custom dataset
  Request: multipart/form-data with JSON file
  Response:
    dataset_id: string
    num_dialogues: int

# Session Endpoints
GET /api/sessions
  Description: List recent chat sessions
  Response:
    sessions: List[SessionSummary]

GET /api/sessions/{session_id}
  Description: Get session details with evaluation
  Response:
    Session with goals and turns
```

### 7.2 WebSocket Protocol

```yaml
# WebSocket: /api/chat

# Client -> Server: Send message
{
  "type": "message",
  "session_id": "string (optional for new session)",
  "content": "User message text"
}

# Server -> Client: Agent response + evaluation
{
  "type": "response",
  "session_id": "string",
  "turn_number": int,
  "agent_response": "Agent's response text",
  
  # Evaluation data (streamed as available)
  "evaluation": {
    "is_new_goal": bool,
    "goal_number": int,
    "turn_quality": "success|failure|pending",
    "rcof": "E1-E7|null",
    "judge_reasoning": "string (think content)",
    "current_gsr": float
  }
}

# Server -> Client: Evaluation update (if streaming)
{
  "type": "evaluation_update",
  "session_id": "string",
  "turn_number": int,
  "evaluation": { ... }
}

# Server -> Client: Error
{
  "type": "error",
  "message": "Error description"
}
```

---

## 8. Frontend Architecture

### 8.1 Component Hierarchy

```
App
├── Layout
│   ├── Header
│   └── Navigation
│
├── Pages
│   ├── ChatPage
│   │   ├── ChatInterface
│   │   │   ├── MessageList
│   │   │   │   └── MessageBubble
│   │   │   ├── InputArea
│   │   │   └── SessionSelector
│   │   │
│   │   └── JudgePanel
│   │       ├── GoalSegmentDisplay
│   │       ├── TurnEvaluationList
│   │       ├── GSRIndicator
│   │       └── ReasoningDisplay
│   │
│   ├── EvaluationPage
│   │   ├── DatasetSelector
│   │   ├── EvaluationProgress
│   │   └── ResultsDisplay
│   │
│   └── DashboardPage
│       ├── GSRChart
│       ├── RCOFDistribution
│       ├── TrendAnalysis
│       └── DomainComparison
│
└── Shared
    ├── LoadingSpinner
    ├── ErrorBoundary
    └── Toast
```

### 8.2 State Management

```typescript
// Using React Context + Hooks for simplicity

interface ChatState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  evaluation: SessionEvaluation | null;
  isLoading: boolean;
}

interface SessionEvaluation {
  goals: GoalEvaluation[];
  currentGSR: number;
  turnCount: number;
}

interface GoalEvaluation {
  goalNumber: number;
  turns: TurnEvaluation[];
  status: 'success' | 'failure' | 'in_progress';
}

interface TurnEvaluation {
  turnNumber: number;
  isNewGoal: boolean;
  quality: 'success' | 'failure' | 'pending';
  rcof: string | null;
  reasoning: string;
}
```

### 8.3 WebSocket Integration

```typescript
// hooks/useChat.ts

function useChat() {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [evaluation, setEvaluation] = useState<SessionEvaluation | null>(null);
  
  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000/api/chat');
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'response':
          // Add agent message
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: data.agent_response,
            evaluation: data.evaluation
          }]);
          // Update evaluation state
          updateEvaluation(data.evaluation);
          break;
          
        case 'evaluation_update':
          // Stream evaluation updates
          updateTurnEvaluation(data);
          break;
          
        case 'error':
          toast.error(data.message);
          break;
      }
    };
    
    setWs(socket);
    return () => socket.close();
  }, []);
  
  const sendMessage = (content: string, sessionId?: string) => {
    ws?.send(JSON.stringify({
      type: 'message',
      session_id: sessionId,
      content
    }));
    
    // Optimistically add user message
    setMessages(prev => [...prev, { role: 'user', content }]);
  };
  
  return { messages, evaluation, sendMessage };
}
```

---

## 9. Dataset Management

### 9.1 MultiWOZ Loader

```python
class MultiWOZLoader:
    """
    Loader for MultiWOZ 2.4 dataset.
    
    Converts MultiWOZ format to MindTheGoal Session format.
    """
    
    DATASET_URL = "https://github.com/budzianowski/multiwoz"
    
    def __init__(self, data_dir: str = "datasets/data/multiwoz"):
        self.data_dir = data_dir
        self.dialogues = {}
    
    async def load(self, sample_size: Optional[int] = None) -> List[Session]:
        """Load and convert MultiWOZ dialogues."""
        
        # Load from Hugging Face datasets or local files
        if not self._data_exists():
            await self._download_dataset()
        
        raw_dialogues = self._load_raw_dialogues()
        
        if sample_size:
            raw_dialogues = random.sample(
                list(raw_dialogues.items()), 
                min(sample_size, len(raw_dialogues))
            )
        else:
            raw_dialogues = list(raw_dialogues.items())
        
        sessions = []
        for dialogue_id, dialogue_data in raw_dialogues:
            session = self._convert_to_session(dialogue_id, dialogue_data)
            sessions.append(session)
        
        return sessions
    
    def _convert_to_session(
        self, 
        dialogue_id: str, 
        dialogue_data: dict
    ) -> Session:
        """Convert MultiWOZ dialogue to Session format."""
        
        turns = []
        for i, log in enumerate(dialogue_data.get("log", [])):
            if i % 2 == 0:  # User turns
                user_msg = log.get("text", "")
            else:  # System turns
                system_msg = log.get("text", "")
                turns.append(Turn(
                    turn_number=len(turns) + 1,
                    user_message=user_msg,
                    agent_response=system_msg
                ))
        
        return Session(
            session_id=dialogue_id,
            goals=[],  # Goals will be segmented during evaluation
            metadata={
                "source": "multiwoz",
                "domains": dialogue_data.get("domains", [])
            }
        )
```

### 9.2 Custom Dataset Format

```json
{
  "dialogues": [
    {
      "dialogue_id": "unique_id",
      "metadata": {
        "domain": "optional",
        "source": "optional"
      },
      "turns": [
        {
          "turn_id": 1,
          "user": "User message",
          "system": "System response"
        }
      ]
    }
  ]
}
```

---

## 10. Evaluation Pipeline

### 10.1 Batch Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Batch Evaluation Pipeline                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Dataset                                                 │
│    • MultiWOZ loader or Custom JSON                             │
│    • Sample if requested                                        │
│    • Convert to Session format                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Process Each Session                                         │
│    ┌───────────────────────────────────────────────────────┐    │
│    │ For each turn in session:                             │    │
│    │   a. Send to Judge Agent                              │    │
│    │   b. Get: is_new_goal, quality, rcof                  │    │
│    │   c. Store evaluation result                          │    │
│    │   d. Update progress                                  │    │
│    └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Segment Goals                                                │
│    • Group turns by is_new_goal boundaries                      │
│    • Assign goal numbers                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Calculate Metrics                                            │
│    • GSR per session                                            │
│    • Overall GSR                                                │
│    • Single-turn vs Multi-turn GSR                              │
│    • RCOF distribution                                          │
│    • Domain-level GSR (if applicable)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Generate Report                                              │
│    • Summary statistics                                         │
│    • Detailed breakdowns                                        │
│    • Export formats: JSON, CSV, HTML                            │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Real-time Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   Real-time Chat + Evaluation                   │
└─────────────────────────────────────────────────────────────────┘

User Input ──► WebSocket ──► FastAPI Handler
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌──────────────┐              ┌──────────────┐
            │  Chat Agent  │              │ Judge Agent  │
            │  (Claude)    │              │ (Claude+CoT) │
            └──────┬───────┘              └──────┬───────┘
                   │                             │
                   │ Response                    │ Evaluation
                   │                             │
                   └───────────┬─────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Combine Results  │
                    │ Update Session   │
                    │ Calculate GSR    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ WebSocket Push   │
                    │ to Frontend      │
                    └──────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │   Chat   │  │  Judge   │  │   GSR    │
        │ Message  │  │  Panel   │  │ Update   │
        └──────────┘  └──────────┘  └──────────┘
```

---

## 11. Error Handling & Resilience

### 11.1 AWS Credential Expiration

```python
async def invoke_with_retry(self, messages, max_retries=3):
    """Invoke LLM with automatic credential refresh."""
    for attempt in range(max_retries):
        try:
            return await self.llm.ainvoke(messages)
        except Exception as e:
            if is_expired_token_error(e) and attempt < max_retries - 1:
                logger.warning(f"Token expired, refreshing (attempt {attempt + 1})")
                self._refresh_credentials()
                continue
            raise
```

### 11.2 Rate Limiting

```python
from asyncio_throttle import Throttler

class RateLimitedClient:
    def __init__(self, requests_per_minute: int = 60):
        self.throttler = Throttler(
            rate_limit=requests_per_minute,
            period=60.0
        )
    
    async def invoke(self, messages):
        async with self.throttler:
            return await self._invoke(messages)
```

### 11.3 Graceful Degradation

- If judge evaluation fails, mark turn as "pending" and continue
- If chat agent fails, return error message to user
- If batch job fails mid-way, save partial results

---

## 12. Security Considerations

### 12.1 AWS Credentials

- Use IAM roles or environment variables
- Never commit credentials to code
- Implement credential rotation

### 12.2 Input Validation

- Sanitize user inputs before LLM calls
- Validate JSON schemas for API requests
- Limit message lengths to prevent abuse

### 12.3 Data Privacy

- Don't log sensitive conversation content
- Implement session expiration
- Provide data deletion capability

---

## 13. Performance Optimization

### 13.1 Concurrent Processing

```python
async def evaluate_batch(sessions: List[Session]) -> List[EvaluationResult]:
    """Process sessions concurrently with semaphore."""
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
    
    async def process_with_limit(session):
        async with semaphore:
            return await evaluate_session(session)
    
    results = await asyncio.gather(
        *[process_with_limit(s) for s in sessions]
    )
    return results
```

### 13.2 Caching

- Cache evaluation results by turn hash
- Cache goal segmentation decisions
- Use Redis for distributed caching if scaling

### 13.3 Streaming Responses

- Stream chat responses token-by-token
- Push evaluation updates incrementally
- Reduce perceived latency

---

## 14. Future Enhancements

### 14.1 Multi-Teacher Ensemble

As described in the paper, use multiple LLM judges with majority voting:

```python
async def evaluate_with_ensemble(turn, judges):
    """Get evaluation from multiple judges, use majority vote."""
    evaluations = await asyncio.gather(
        *[judge.evaluate(turn) for judge in judges]
    )
    
    # Majority voting
    qualities = [e.quality for e in evaluations]
    majority_quality = Counter(qualities).most_common(1)[0][0]
    
    return ConsensusEvaluation(
        quality=majority_quality,
        agreement=qualities.count(majority_quality) / len(qualities),
        individual_evaluations=evaluations
    )
```

### 14.2 Student Model Distillation

Train a smaller model on teacher labels for faster evaluation:

1. Collect teacher labels from batch evaluation
2. Fine-tune a smaller model (e.g., DistilBERT)
3. Use student for real-time, teacher for validation

### 14.3 Advanced Goal Modeling

- Support non-contiguous goals (graph structure)
- Detect goal dependencies and co-references
- Handle interleaved goals

### 14.4 Hallucination Detection

- Integrate external knowledge base for fact-checking
- Track citations/sources in responses
- Flag responses without grounding

---

## Appendix A: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS
export AWS_REGION=us-west-2
export AWS_PROFILE=default  # or your profile

# Or use .env file
cp .env.example .env
# Edit .env with your settings

# Run backend
python run_web.py

# Run frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## Appendix B: API Examples

### Start Evaluation

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"dataset": "multiwoz", "sample_size": 100}'
```

### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/api/chat');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'message',
    content: 'I need to book a hotel in Cambridge'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent:', data.agent_response);
  console.log('Evaluation:', data.evaluation);
};
```

---

*Document Version: 1.0 | Last Updated: January 2026*
