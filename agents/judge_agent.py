"""Judge Agent for evaluating conversation turns."""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from agents.bedrock_client import BedrockClient
from core.models import Turn, TurnQuality, TurnEvaluation, Session
from core.rcof_classifier import RCOFClassifier
from config import get_settings

logger = logging.getLogger(__name__)


# Improved system prompt that emphasizes goal boundary detection
JUDGE_SYSTEM_PROMPT = """You are an expert conversation evaluator. Your task is to evaluate each turn in a multi-turn conversation between a user and an AI assistant.

For each turn, you must determine:

1. **Is New Goal**: Does this turn start a NEW user goal/intent?
   
   MARK AS NEW GOAL (is_new_goal: true) when:
   - User switches to a DIFFERENT topic or domain (e.g., from restaurant to hotel)
   - User introduces a NEW request unrelated to previous discussion
   - User says things like "I also need...", "Another thing...", "Now I want...", "Can you also..."
   - User asks about a completely different service/entity
   
   KEEP AS SAME GOAL (is_new_goal: false) when:
   - User asks follow-up questions about the SAME topic
   - User requests clarification or more details on SAME topic
   - User modifies constraints for the SAME request (e.g., "actually make it 3 people instead of 2")
   
   IMPORTANT: A single conversation often contains 2-5 DIFFERENT goals! 
   In task-oriented dialogues (restaurant, hotel, taxi, train, attractions), users typically:
   - Book a restaurant (Goal 1)
   - Then book a hotel nearby (Goal 2) 
   - Then book a taxi (Goal 3)
   
   You MUST identify these goal boundaries accurately!

2. **Quality**: Was the agent's response successful or a failure?
   - SUCCESS: The agent's response adequately addresses the user's request
   - FAILURE: The agent's response fails to address the user's needs

3. **Root Cause of Failure (RCOF)**: If the turn is a failure, classify:
   - E1: Language Understanding Failure - Misunderstood the user's request
   - E2: Refusal to Answer - Inappropriately refused despite ability to help  
   - E3: Incorrect Retrieval - Retrieved wrong information
   - E4: Retrieval Failure - Failed to retrieve relevant information
   - E5: System Error - Technical issues (timeout, truncation, etc.)
   - E6: Incorrect Routing - Routed to wrong domain/module
   - E7: Out-of-Domain - Request outside system's scope

4. **Reasoning**: Brief explanation including:
   - Why you marked this as new goal or not
   - Why you marked success/failure

Respond ONLY with a valid JSON array. No other text."""


EVALUATION_PROMPT_TEMPLATE = """Evaluate this conversation. Pay special attention to detecting when the user switches to a NEW goal/topic.

CONVERSATION:
{conversation}

Return a JSON array with one object per turn:

```json
[
  {{
    "turn_number": 1,
    "is_new_goal": true,
    "quality": "success",
    "rcof": null,
    "reasoning": "First turn - user asks about restaurants. This is Goal 1."
  }},
  {{
    "turn_number": 2,
    "is_new_goal": false,
    "quality": "success",
    "rcof": null,
    "reasoning": "Still discussing restaurants (same goal). Agent provides info."
  }},
  {{
    "turn_number": 3,
    "is_new_goal": true,
    "quality": "success",
    "rcof": null,
    "reasoning": "USER SWITCHES TO HOTEL - this is Goal 2. Different service/domain."
  }}
]
```

CRITICAL: Look for domain/topic switches. Users often have 2-5 goals per conversation!
Return ONLY the JSON array, no other text."""


class JudgeAgent:
    """
    LLM-based judge for evaluating conversation quality.
    
    Implements the evaluation methodology from the MindTheGoal paper:
    - Turn-level success/failure classification
    - Goal segmentation (detecting new goals)
    - Root Cause of Failure (RCOF) attribution
    """
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        temperature: Optional[float] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize Judge Agent.
        
        Args:
            bedrock_client: Optional pre-configured Bedrock client
            temperature: Temperature for evaluations (low for consistency)
            log_dir: Directory for detailed evaluation logs
        """
        settings = get_settings()
        
        self.temperature = temperature or settings.judge_temperature
        self.log_dir = Path(log_dir) if log_dir else Path("logs/evaluations")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if bedrock_client:
            self.client = bedrock_client
        else:
            self.client = BedrockClient(temperature=self.temperature)
        
        self.rcof_classifier = RCOFClassifier()
    
    async def evaluate_turn(
        self,
        turn: Turn,
        conversation_context: List[Turn]
    ) -> TurnEvaluation:
        """
        Evaluate a single turn with context.
        
        Args:
            turn: The turn to evaluate
            conversation_context: Previous turns for context
            
        Returns:
            TurnEvaluation with quality and RCOF
        """
        # Build conversation text
        conversation_text = self._format_conversation(conversation_context + [turn])
        
        prompt = EVALUATION_PROMPT_TEMPLATE.format(conversation=conversation_text)
        
        response = await self.client.invoke_with_json_output(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=self.temperature
        )
        
        # Parse response
        if isinstance(response, list) and len(response) > 0:
            eval_data = response[-1]  # Get last turn's evaluation
        elif isinstance(response, dict) and "raw_response" not in response:
            eval_data = response
        else:
            # Fallback if parsing failed
            logger.warning("Failed to parse judge response, using defaults")
            eval_data = {
                "turn_number": turn.turn_number,
                "is_new_goal": turn.turn_number == 1,
                "quality": "success",
                "rcof": None,
                "reasoning": "Parse error - defaulting to success"
            }
        
        quality = TurnQuality.SUCCESS if eval_data.get("quality") == "success" else TurnQuality.FAILURE
        
        return TurnEvaluation(
            turn_number=eval_data.get("turn_number", turn.turn_number),
            is_new_goal=eval_data.get("is_new_goal", turn.turn_number == 1),
            quality=quality,
            rcof=eval_data.get("rcof") if quality == TurnQuality.FAILURE else None,
            reasoning=eval_data.get("reasoning", "")
        )
    
    async def evaluate_session(
        self, 
        session: Session,
        session_index: Optional[int] = None
    ) -> List[TurnEvaluation]:
        """
        Evaluate all turns in a session.
        
        Args:
            session: Session containing turns to evaluate
            session_index: Optional index for logging purposes
            
        Returns:
            List of TurnEvaluations
        """
        turns = session.turns
        
        if not turns:
            return []
        
        # Build full conversation text
        conversation_text = self._format_conversation(turns)
        
        prompt = EVALUATION_PROMPT_TEMPLATE.format(conversation=conversation_text)
        
        response = await self.client.invoke_with_json_output(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=self.temperature
        )
        
        evaluations = []
        raw_response = response  # Keep for logging
        
        if isinstance(response, list):
            for i, eval_data in enumerate(response):
                turn_num = eval_data.get("turn_number", i + 1)
                quality = TurnQuality.SUCCESS if eval_data.get("quality") == "success" else TurnQuality.FAILURE
                
                evaluations.append(TurnEvaluation(
                    turn_number=turn_num,
                    is_new_goal=eval_data.get("is_new_goal", i == 0),
                    quality=quality,
                    rcof=eval_data.get("rcof") if quality == TurnQuality.FAILURE else None,
                    reasoning=eval_data.get("reasoning", "")
                ))
        else:
            # Fallback: mark all as success
            logger.warning(f"Failed to parse response for session {session.session_id}, defaulting to success")
            for i, turn in enumerate(turns):
                evaluations.append(TurnEvaluation(
                    turn_number=i + 1,
                    is_new_goal=i == 0,
                    quality=TurnQuality.SUCCESS,
                    rcof=None,
                    reasoning="Evaluation parse error - defaulting to success"
                ))
        
        # Log detailed evaluation
        self._log_evaluation(session, evaluations, raw_response, session_index)
        
        return evaluations
    
    def _log_evaluation(
        self,
        session: Session,
        evaluations: List[TurnEvaluation],
        raw_response: Any,
        session_index: Optional[int] = None
    ):
        """Log detailed evaluation results for a session."""
        log_entry = {
            "session_id": session.session_id,
            "session_index": session_index,
            "num_turns": len(session.turns),
            "num_goals_detected": sum(1 for e in evaluations if e.is_new_goal),
            "num_failures": sum(1 for e in evaluations if e.quality == TurnQuality.FAILURE),
            "turns": [],
            "raw_llm_response": raw_response if isinstance(raw_response, (list, dict)) else str(raw_response)
        }
        
        for turn, eval_result in zip(session.turns, evaluations):
            log_entry["turns"].append({
                "turn_number": turn.turn_number,
                "user_message": turn.user_message[:200] + "..." if len(turn.user_message) > 200 else turn.user_message,
                "agent_response": turn.agent_response[:200] + "..." if len(turn.agent_response) > 200 else turn.agent_response,
                "is_new_goal": eval_result.is_new_goal,
                "quality": eval_result.quality.value,
                "rcof": eval_result.rcof,
                "reasoning": eval_result.reasoning
            })
        
        # Log to file
        log_file = self.log_dir / f"session_{session_index or session.session_id}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)
        
        # Also log summary to logger
        goals = log_entry["num_goals_detected"]
        failures = log_entry["num_failures"]
        logger.info(
            f"Session {session_index}: {len(session.turns)} turns, "
            f"{goals} goals detected, {failures} failures"
        )
    
    async def apply_evaluations(
        self,
        session: Session,
        evaluations: List[TurnEvaluation]
    ) -> Session:
        """
        Apply evaluations to session turns.
        
        Args:
            session: Session to update
            evaluations: Turn evaluations to apply
            
        Returns:
            Updated session with evaluated turns
        """
        eval_map = {e.turn_number: e for e in evaluations}
        
        for turn in session.turns:
            if turn.turn_number in eval_map:
                eval_result = eval_map[turn.turn_number]
                turn.quality = eval_result.quality
                turn.rcof = eval_result.rcof
                turn.is_new_goal = eval_result.is_new_goal
                turn.judge_reasoning = eval_result.reasoning
        
        return session
    
    def _format_conversation(self, turns: List[Turn]) -> str:
        """Format turns into readable conversation text."""
        lines = []
        for turn in turns:
            lines.append(f"[Turn {turn.turn_number}]")
            lines.append(f"USER: {turn.user_message}")
            lines.append(f"AGENT: {turn.agent_response}")
            lines.append("")
        return "\n".join(lines)
    
    async def evaluate_single_response(
        self,
        user_message: str,
        agent_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Quick evaluation of a single response (for real-time chat).
        
        Args:
            user_message: Current user message
            agent_response: Agent's response to evaluate
            conversation_history: Previous messages as list of dicts
            
        Returns:
            Evaluation result dictionary
        """
        # Build context
        context_turns = []
        if conversation_history:
            for i, msg in enumerate(conversation_history):
                context_turns.append(Turn(
                    turn_number=i + 1,
                    user_message=msg.get("user", ""),
                    agent_response=msg.get("agent", "")
                ))
        
        # Add current turn
        current_turn = Turn(
            turn_number=len(context_turns) + 1,
            user_message=user_message,
            agent_response=agent_response
        )
        
        evaluation = await self.evaluate_turn(current_turn, context_turns)
        
        return {
            "quality": evaluation.quality.value,
            "is_new_goal": evaluation.is_new_goal,
            "rcof": evaluation.rcof,
            "reasoning": evaluation.reasoning
        }
