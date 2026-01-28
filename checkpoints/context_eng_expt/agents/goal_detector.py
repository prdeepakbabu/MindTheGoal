"""Goal detector wrapper using MindTheGoal's goal segmentation."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')
from agents.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


GOAL_DETECTION_PROMPT = """Analyze this conversation turn to determine if the user is starting a NEW goal or continuing their CURRENT goal.

A NEW goal means:
- User explicitly mentions a different topic/task
- User says "also", "another thing", "next I need"
- User's request is unrelated to the previous turns
- Previous task appears complete and user starts something else

CONTINUING current goal means:
- User provides additional details for the same request
- User answers a clarifying question
- User confirms/modifies the current booking/task
- User is still working on the same overall objective

Conversation history:
{history}

Current user message:
{current_message}

Respond with ONLY one word: NEW or CONTINUE"""


@dataclass
class GoalDetector:
    """
    Detects goal boundaries in conversations.
    
    Uses LLM to determine if a user message starts a new goal
    or continues the current goal.
    """
    
    _llm: Optional[BedrockClient] = field(default=None, init=False)
    _history: List[Dict[str, str]] = field(default_factory=list, init=False)
    _goal_count: int = field(default=0, init=False)
    _goal_boundaries: List[int] = field(default_factory=list, init=False)  # Turn indices where new goals start
    
    def __post_init__(self):
        """Initialize the LLM client."""
        self._llm = BedrockClient()
    
    async def is_new_goal(self, user_message: str, agent_response: Optional[str] = None) -> bool:
        """
        Determine if the user message starts a new goal.
        
        Args:
            user_message: The user's message to analyze
            agent_response: The previous agent response (if any)
            
        Returns:
            True if this message starts a new goal
        """
        # First message is always a new goal
        if not self._history:
            self._goal_count = 1
            self._goal_boundaries.append(0)
            self._update_history(user_message, agent_response)
            return True
        
        # Build prompt
        history_text = self._format_history()
        prompt = GOAL_DETECTION_PROMPT.format(
            history=history_text,
            current_message=user_message
        )
        
        try:
            response = await self._llm.invoke(prompt, temperature=0.1)
            is_new = response.strip().upper().startswith("NEW")
            
            if is_new:
                self._goal_count += 1
                self._goal_boundaries.append(len(self._history) // 2)  # Turn index
                logger.info(f"New goal detected (goal #{self._goal_count})")
            
            self._update_history(user_message, agent_response)
            return is_new
            
        except Exception as e:
            logger.error(f"Error detecting goal boundary: {e}")
            # Default to continuing current goal on error
            self._update_history(user_message, agent_response)
            return False
    
    def _update_history(self, user_message: str, agent_response: Optional[str]):
        """Update internal history with the latest turn."""
        self._history.append({"role": "user", "content": user_message})
        if agent_response:
            self._history.append({"role": "assistant", "content": agent_response})
    
    def _format_history(self) -> str:
        """Format history for the detection prompt."""
        if not self._history:
            return "[No previous conversation]"
        
        # Keep last few turns for context (avoid huge prompts)
        recent = self._history[-10:]
        lines = []
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)
    
    def add_agent_response(self, response: str):
        """Add agent response to history (for cases where we detect before response)."""
        if self._history and self._history[-1]["role"] == "user":
            self._history.append({"role": "assistant", "content": response})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get goal detection statistics."""
        return {
            "total_goals": self._goal_count,
            "goal_boundaries": self._goal_boundaries,
            "total_turns": len(self._history) // 2
        }
    
    def reset(self):
        """Reset for a new conversation."""
        self._history = []
        self._goal_count = 0
        self._goal_boundaries = []
