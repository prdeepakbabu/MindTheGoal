"""Simulated user agent that plays a user role with predefined goals."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')
from agents.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


@dataclass
class UserGoal:
    """Represents a user goal in the simulation."""
    description: str
    achieved: bool = False
    turns_spent: int = 0


@dataclass
class SimulatedUser:
    """
    LLM-based simulated user that plays a user role with given goals and persona.
    
    The simulated user generates realistic utterances based on:
    - A list of goals to accomplish (in order)
    - A persona that defines communication style
    - The conversation history so far
    """
    
    goals: List[str]
    persona: str
    persona_modifier: str = ""
    
    # Internal state
    current_goal_idx: int = field(default=0, init=False)
    conversation_history: List[Dict[str, str]] = field(default_factory=list, init=False)
    _llm: Optional[BedrockClient] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the LLM client."""
        self._llm = BedrockClient()
        self._goal_states = [UserGoal(description=g) for g in self.goals]
    
    @property
    def current_goal(self) -> Optional[str]:
        """Get the current goal the user is pursuing."""
        if self.current_goal_idx < len(self.goals):
            return self.goals[self.current_goal_idx]
        return None
    
    @property
    def all_goals_achieved(self) -> bool:
        """Check if all goals have been achieved."""
        return self.current_goal_idx >= len(self.goals)
    
    def mark_current_goal_achieved(self):
        """Mark the current goal as achieved and move to next."""
        if self.current_goal_idx < len(self._goal_states):
            self._goal_states[self.current_goal_idx].achieved = True
            self.current_goal_idx += 1
            logger.info(f"Goal achieved, moving to goal {self.current_goal_idx + 1}")
    
    async def generate_utterance(self, agent_response: Optional[str] = None) -> str:
        """
        Generate the next user utterance based on goals and conversation history.
        
        Args:
            agent_response: The chatbot's last response (None for first turn)
            
        Returns:
            The user's next message
        """
        # Update conversation history
        if agent_response:
            self.conversation_history.append({
                "role": "assistant",
                "content": agent_response
            })
        
        # Build the prompt for the simulated user
        prompt = self._build_user_prompt()
        
        # Generate response
        try:
            response = await self._llm.invoke(prompt, temperature=0.8)
            user_message = self._extract_user_message(response)
            
            # Add to history
            self.conversation_history.append({
                "role": "user", 
                "content": user_message
            })
            
            # Track turns for current goal
            if self.current_goal_idx < len(self._goal_states):
                self._goal_states[self.current_goal_idx].turns_spent += 1
            
            return user_message
            
        except Exception as e:
            logger.error(f"Error generating user utterance: {e}")
            raise
    
    def _build_user_prompt(self) -> str:
        """Build the prompt for generating the user's next message."""
        goals_text = "\n".join([
            f"{i+1}. {'✓ ' if self._goal_states[i].achieved else ''}{g}"
            for i, g in enumerate(self.goals)
        ])
        
        history_text = self._format_history()
        
        prompt = f"""You are role-playing as a user interacting with a helpful assistant.

## Your Persona
{self.persona}
{self.persona_modifier}

## Your Goals (complete them in order)
{goals_text}

## Current Goal
{self.current_goal if self.current_goal else "All goals completed - say goodbye and end the conversation."}

## Conversation So Far
{history_text if history_text else "[This is the start of the conversation]"}

## Instructions
- Generate ONLY the user's next message (1-3 sentences)
- Stay in character with your persona
- Work toward achieving your current goal
- If the assistant has just helped you complete your current goal, acknowledge it and move to the next goal
- If all goals are complete, thank the assistant and indicate you're done
- Be natural and conversational
- Do NOT include any meta-commentary or labels like "User:"

## Your Response (user's next message only):"""
        
        return prompt
    
    def _format_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return ""
        
        lines = []
        for turn in self.conversation_history:
            role = "You" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        
        return "\n".join(lines)
    
    def _extract_user_message(self, response: str) -> str:
        """Extract clean user message from LLM response."""
        # Remove common prefixes that the LLM might add
        prefixes_to_remove = [
            "User:", "user:", "You:", "you:", 
            "Response:", "response:",
            "Message:", "message:"
        ]
        
        message = response.strip()
        for prefix in prefixes_to_remove:
            if message.startswith(prefix):
                message = message[len(prefix):].strip()
        
        # Remove quotes if the whole message is quoted
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]
        
        return message
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the simulated user."""
        return {
            "goals": self.goals,
            "current_goal_idx": self.current_goal_idx,
            "current_goal": self.current_goal,
            "all_goals_achieved": self.all_goals_achieved,
            "goal_states": [
                {
                    "description": gs.description,
                    "achieved": gs.achieved,
                    "turns_spent": gs.turns_spent
                }
                for gs in self._goal_states
            ],
            "total_turns": len([h for h in self.conversation_history if h["role"] == "user"]),
            "persona": self.persona
        }
    
    def should_end_conversation(self, agent_response: str) -> bool:
        """
        Determine if the conversation should end based on context.
        
        This is a heuristic check - ideally the user LLM will naturally
        end the conversation when goals are achieved.
        """
        # Check for goodbye signals in the agent's response after all goals done
        if self.all_goals_achieved:
            goodbye_signals = ["goodbye", "bye", "have a nice", "take care", "anything else"]
            response_lower = agent_response.lower()
            return any(signal in response_lower for signal in goodbye_signals)
        
        return False
