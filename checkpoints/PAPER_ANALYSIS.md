# Detailed Analysis: "Mind the Goal: Data-Efficient Goal-Oriented Evaluation of Conversational Agents and Chatbots using Teacher Models"

**arXiv: 2510.03696v1** | **Submitted: October 4, 2025**

**Authors:** Deepak Babu Piskala, Sharlene Chen, Udita Patel, Parul Kalra, Rafael Castrillo  
**Affiliation:** Amazon.com, Seattle, WA, USA

---

## Executive Summary

This paper presents **"Mind the Goal" (MindTheGoal)**, a comprehensive framework for evaluating multi-turn conversational agents by focusing on **goal-level success** rather than traditional turn-level metrics. The framework introduces two key metrics: **Goal Success Rate (GSR)** and **Root Cause of Failure (RCOF)** taxonomy. Applied to AIDA (an enterprise employee conversational agent at Amazon), the framework demonstrated GSR improvement from **63% to 79%** over six months.

---

## 1. Problem Statement & Motivation

### The Core Problem
Traditional chatbot evaluation methods suffer from significant limitations:

1. **Turn-level focus**: Most existing methods assess interactions at individual turn levels (user query + bot response) without addressing whether the user's overarching goal was fulfilled.

2. **Incomplete picture**: Metrics like BLEU, ROUGE, and BERTScore provide useful signals but fail to capture:
   - Whether the user's underlying goal was achieved
   - The end-to-end conversational success
   - Where in a multi-turn exchange the assistant failed

3. **Multi-turn complexity**: Analysis of AIDA showed:
   - **39% of dialogs involve multiple turns**
   - Multi-turn sessions exhibit **~3x higher negative feedback rate** (2.65% vs 0.9%) compared to single-turn sessions

### Why This Matters
Modern conversational assistants use agentic LLM architectures with:
- Tool invocation
- External memory reads/writes
- Multi-step planning
- Multi-agent collaboration (HR, IT, legal, analytics agents)

Each new component (memory layer, tool wrapper, inter-agent message channel) compounds the risk of **subtle cascading failures**, making robust evaluation critical.

---

## 2. Key Definitions & Framework

### 2.1 Conversation Anatomy

| Unit | Definition |
|------|------------|
| **Session (S)** | Full interaction between user and chatbot, bounded by timeout or user exit |
| **Goal (Gᵢ)** | Coherent user intent or information need (e.g., "Where can I submit expenses?") |
| **Turn (Tⱼ)** | User query (qⱼ) + chatbot response (rⱼ) |

**Hierarchy:** Session → Goals → Turns

### 2.2 Goal Success Criteria
A goal is marked **successful** only if **ALL its turns are successful**—this is deliberately strict to ensure high fidelity to user experience.

```
GoalQuality(Gₖ) = 
  success, if ∀ Tⱼ ∈ Gₖ, quality(Tⱼ) = success
  failure, otherwise
```

---

## 3. The MindTheGoal Framework (CIM - Conversational Intelligence Model)

The framework consists of three main components:

### 3.1 Goal Segmentation
**Purpose:** Identify boundaries between distinct goals within a dialog

**Approach:**
- For each turn Tⱼ, predict `is_new_goal(Tⱼ) ∈ {yes, no}`
- First turn is always a new goal
- Features: lexical cues, contextual cues, temporal gaps

**Output:** Divides N turns into K goals (G₁, G₂, ..., Gₖ)

### 3.2 Goal Success Rate (GSR)
**Formula:**
```
GSR = (1/K) × Σₖ 𝟙[goalQuality(Gₖ) = success] × 100%
```

**Key Insight:** Even if user eventually gets answer after rephrasing, the goal is still marked **failed** because the user had to work through a failed response.

### 3.3 Root Cause of Failure (RCOF) Taxonomy

Seven distinct error categories for failed goals:

| Code | Category | Description |
|------|----------|-------------|
| **E1** | Language Understanding Failure | Misunderstood user's request or context |
| **E2** | Refusal to Answer | Inappropriate refusal despite being able to help |
| **E3** | Incorrect Retrieval | Retrieved wrong informational content (RAG issue) |
| **E4** | Retrieval Failure | Failed to retrieve any relevant information |
| **E5** | System Error | Technical issues (timeout, truncation, integration failure) |
| **E6** | Incorrect Routing | Query routed to wrong domain/module |
| **E7** | Out-of-Domain Query | Request outside system's designed scope |

**Attribution Rule:** Goal's RCOF = error category of its **earliest failed turn** (assumption: initial breakdowns are most disruptive)

---

## 4. Methodology: Human-in-the-Loop (HITL) Pipeline

### 4.1 Pipeline Overview

```
AIDA Event Logs → Preprocessing → Multi-Teacher Evaluation → Majority Voting → Ground Truth Labels
                                                               ↓
                                            (Disagreements escalate to human experts)
```

### 4.2 Multi-Teacher Supervision
- **Three independent expert teacher models** evaluate each conversation
- Models used: Claude Sonnet, Claude Haiku, GPT-4, LLaMA-4
- **Chain-of-Thought (CoT) prompting** with explicit reasoning tags (`<think>...</think>`)

### 4.3 Label Aggregation
- **Majority voting:** If 2+ teacher models agree → accept as ground truth
- **All disagree:** Mark as ambiguous → escalate to human annotators (using SOPs)

### 4.4 Student Model Distillation
- Optional: Distill teacher ensemble into lightweight student model
- Enables efficient real-time and offline inference at lower cost

---

## 5. Experimental Setup: AIDA System

### 5.1 System Description
- **AIDA:** Enterprise-grade virtual assistant at Amazon
- **Deployment:** Desktop and mobile platforms
- **Purpose:** Help employees with workplace queries

### 5.2 Capabilities
- HR policies, IT troubleshooting, expense reimbursement
- Time-off requests, internal tools/documentation access
- **Action-oriented goals:** Leave applications, meeting room bookings

### 5.3 Dataset
- **~10,000 multi-turn conversations** over 30 days
- Rich annotations including:
  - Implicit signals (rephrases, abandonments, search fallback)
  - Explicit feedback (likes, thumbs up/down)
  - Metadata (device type, timestamp, citations)

---

## 6. Results & Findings

### 6.1 Overall Performance

| Metric | Value |
|--------|-------|
| **Overall GSR** | 78% |
| **Multi-turn GSR** | 66% |
| Total Goals (sample) | 1,915 |
| Successful Goals | 1,488 (77.7%) |
| Failed Goals | 427 (22.3%) |

**Key Finding:** Conversational complexity (multi-turn) significantly increases failure risk.

### 6.2 Failure Root Cause Distribution

| Root Cause | Count | % of Goals |
|------------|-------|------------|
| Retrieval Failure (E4) | 164 | 8.6% |
| Language Understanding (E1) | 116 | 6.1% |
| Incorrect Retrieval (E3) | 70 | 3.7% |
| System Error (E5) | 43 | 2.2% |
| Refusal (E2) | 17 | 0.9% |

**Top 3 failure causes:**
1. **Retrieval failures** (39% of failures)
2. **Language understanding errors** (27%)
3. **Incorrect retrievals** (16%)

### 6.3 Temporal Improvement
Over six months (Oct 2024 → May 2025):
- GSR improved from **63% → 79%** (16 percentage point increase)
- Multi-turn GSR rose by **12 points**

### 6.4 Improvement Drivers
Not driven by prompt tuning or fallback rules, but by:
- Improved source integration
- Better routing mechanisms
- Upgraded models with better language reasoning
- Agentic behaviors (clarification questions, multi-source synthesis)

### 6.5 Human-LLM Agreement
- **75% agreement** between human annotators and teacher model ensemble
- Task-specific disagreement:
  - Goal segmentation/turn quality: 13% disagreement
  - RCOF attribution: 17% disagreement
- Target: Reduce discrepancies to **<5%** through feedback integration

---

## 7. Key Contributions

1. **General Goal-Oriented Framework:** Segments dialogs by user goals and evaluates success at goal level (not turn level)

2. **GSR Metric:** Quantifies fraction of satisfied user goals with strict success criteria

3. **RCOF Taxonomy:** Categorizes and explains failed goals with actionable error categories

4. **Model-Based Implementation:** Demonstrates LLM teacher models can assist in labeling at scale

5. **Empirical Validation:** Shows framework captures holistic quality signals and drives real system improvements

---

## 8. Limitations & Future Directions

### Current Limitations

1. **Scope:** Best suited for task-oriented dialogs with clear success criteria; struggles with open-ended scenarios (document summarization, email composition)

2. **Contiguity Assumption:** Assumes each goal corresponds to contiguous turns; doesn't support:
   - Non-consecutive goal spans
   - Interleaved goals
   - Returning to prior topics

3. **Hallucination Detection:** May understate fluent but factually incorrect responses without external verification

### Future Directions
- Model dialog goals as **graph structures** for cross-references and interleaved subgoals
- Alternative formulations:
  - Fractional credit based on successful turn count
  - Goal success based on final turn alone
- Better hallucination detection mechanisms

---

## 9. Technical Implementation Details

### 9.1 LLM Prompt Template Structure

```python
system_prompt = "You are a helpful AI assistant acting as a judge..."

output_format = """
{
  dialog_id: xx,
  turns: [
    {turn_number: 1, is_new_goal: yes/no, quality: success/failure, rcof: E1-E7 | null},
    ...
  ]
}
"""

# Uses <think>...</think> tags for Chain-of-Thought reasoning
```

### 9.2 JSON Schema for Annotations
- `dialog_id`: UUID for joining logs and annotations
- `turn_number`: Sequential index (starts at 1)
- `user_msg` / `response`: Raw text
- `source_urls`: Retrieved document IDs/URLs
- `source_names`: Human-friendly titles
- `source_snippets`: Evidence snippets (≤256 chars)

---

## 10. Comparison to Related Work

| Approach | Level | Goal Awareness | Failure Diagnosis |
|----------|-------|----------------|-------------------|
| BLEU/ROUGE | Turn | ❌ | ❌ |
| BERTScore | Turn | ❌ | ❌ |
| G-Eval | Turn | ❌ | ❌ |
| MT-Bench | Turn | ❌ | ❌ |
| Task Success (Lu et al.) | Task | ✅ (predefined) | ❌ |
| **MindTheGoal (this paper)** | **Goal** | **✅ (inferred)** | **✅ (RCOF)** |

**Key Differentiator:** MindTheGoal extends task success metrics to open-ended, user-driven conversations where goals must be **inferred** and may **evolve**.

---

## 11. Practical Implications

### For Chatbot Developers
1. **Focus on retrieval:** 39% of failures stem from retrieval issues
2. **Improve language understanding:** Second largest failure category
3. **Monitor multi-turn sessions:** Higher complexity = higher failure risk

### For Evaluation Design
1. **Move beyond turn-level metrics** to capture actual user goal fulfillment
2. **Use multi-teacher consensus** for reliable automated labeling
3. **Implement RCOF tracking** for actionable debugging insights

### For Enterprise Deployments
1. **Track GSR over time** as key quality metric
2. **Use RCOF distribution** to prioritize engineering efforts
3. **Leverage HITL pipeline** for scalable quality monitoring

---

## 12. Conclusion

The MindTheGoal framework represents a significant advancement in conversational AI evaluation by:

1. **Shifting focus** from turn-level to goal-level assessment
2. **Providing actionable diagnostics** through RCOF taxonomy
3. **Enabling data-efficient evaluation** via teacher model ensembles
4. **Demonstrating practical value** with 16 percentage point GSR improvement in production system

The framework is **generic** and applicable beyond the AIDA use case, offering a blueprint for any organization seeking to rigorously evaluate and improve multi-turn conversational agents.

---

*Analysis completed: January 23, 2026*
