"""Goal summarizer using LLM to compress completed goals."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')
from agents.bedrock_client import BedrockClient
from .strategies import Turn

logger = logging.getLogger(__name__)


SUMMARIZATION_PROMPT = """Summarize this completed conversation goal in 1-2 sentences.
Focus on: what the user wanted, what was achieved, and key details (names, times, numbers).

Conversation turns for this goal:
{turns}

Write a concise summary (max 50 words):"""


@dataclass
class GoalSummarizer:
    """
    LLM-based summarizer that compresses goal conversation turns into concise summaries.
    """
    
    _llm: Optional[BedrockClient] = field(default=None, init=False)
    _summaries_generated: int = field(default=0, init=False)
    _total_input_tokens: int = field(default=0, init=False)
    _total_output_tokens: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize the LLM client."""
        self._llm = BedrockClient()
    
    async def summarize(self, turns: List[Turn]) -> str:
        """
        Summarize a list of turns from a completed goal.
        
        Args:
            turns: List of Turn objects representing the goal's conversation
            
        Returns:
            A concise summary string
        """
        if not turns:
            return "Empty goal"
        
        # Format turns for the prompt
        turns_text = self._format_turns(turns)
        prompt = SUMMARIZATION_PROMPT.format(turns=turns_text)
        
        # Track input tokens
        self._total_input_tokens += len(prompt) // 4
        
        try:
            summary = await self._llm.invoke(prompt, temperature=0.3)
            summary = self._clean_summary(summary)
            
            # Track output tokens
            self._total_output_tokens += len(summary) // 4
            self._summaries_generated += 1
            
            logger.info(f"Generated summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to simple summary
            return self._fallback_summary(turns)
    
    def _format_turns(self, turns: List[Turn]) -> str:
        """Format turns for the summarization prompt."""
        lines = []
        for turn in turns:
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)
    
    def _clean_summary(self, summary: str) -> str:
        """Clean up the LLM-generated summary."""
        # Remove common prefixes
        prefixes = ["Summary:", "summary:", "Here's the summary:", "The summary is:"]
        for prefix in prefixes:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
        
        # Remove quotes if wrapped
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        
        return summary.strip()
    
    def _fallback_summary(self, turns: List[Turn]) -> str:
        """Generate a simple fallback summary without LLM."""
        # Find first user message as the goal
        first_user = next((t for t in turns if t.role == "user"), None)
        if first_user:
            content = first_user.content[:100]
            return f"User requested: {content}..."
        return "Goal conversation completed"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summarization statistics."""
        return {
            "summaries_generated": self._summaries_generated,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_summary_tokens": (
                self._total_output_tokens / self._summaries_generated
                if self._summaries_generated > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self._summaries_generated = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
