"""Context compaction strategies for the experiment."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .goal_summarizer import GoalSummarizer

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    goal_id: int = 0  # Which goal this turn belongs to


class ContextStrategy(ABC):
    """Base class for context management strategies."""
    
    @abstractmethod
    async def on_new_turn(self, role: str, content: str, is_new_goal: bool = False):
        """Process a new turn in the conversation."""
        pass
    
    @abstractmethod
    def get_context(self) -> str:
        """Get the current context string for the LLM."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the strategy for a new conversation."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the strategy's behavior."""
        pass


@dataclass
class FullContextStrategy(ContextStrategy):
    """
    Baseline strategy: Keep all turns in context without any compression.
    """
    
    turns: List[Turn] = field(default_factory=list)
    
    async def on_new_turn(self, role: str, content: str, is_new_goal: bool = False):
        """Add turn to history."""
        self.turns.append(Turn(role=role, content=content))
    
    def get_context(self) -> str:
        """Return full conversation history."""
        lines = []
        for turn in self.turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)
    
    def reset(self):
        """Clear all turns."""
        self.turns = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "strategy": "full_context",
            "total_turns": len(self.turns),
            "compression_applied": False
        }


@dataclass
class SlidingWindowStrategy(ContextStrategy):
    """
    Keep only the last N turns regardless of goal boundaries.
    """
    
    window_size: int = 10
    turns: List[Turn] = field(default_factory=list)
    
    async def on_new_turn(self, role: str, content: str, is_new_goal: bool = False):
        """Add turn, keeping only window_size most recent."""
        self.turns.append(Turn(role=role, content=content))
    
    def get_context(self) -> str:
        """Return only recent turns within window."""
        recent = self.turns[-self.window_size:] if len(self.turns) > self.window_size else self.turns
        lines = []
        for turn in recent:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)
    
    def reset(self):
        """Clear all turns."""
        self.turns = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "strategy": "sliding_window",
            "window_size": self.window_size,
            "total_turns": len(self.turns),
            "turns_in_context": min(len(self.turns), self.window_size),
            "turns_dropped": max(0, len(self.turns) - self.window_size)
        }


@dataclass
class GoalBoundaryStrategy(ContextStrategy):
    """
    Compress completed goals into summaries when a new goal starts.
    This is the core strategy being tested.
    """
    
    summarizer: Optional['GoalSummarizer'] = None
    
    # Internal state
    goal_summaries: List[str] = field(default_factory=list)
    current_goal_turns: List[Turn] = field(default_factory=list)
    current_goal_id: int = field(default=0)
    _compression_events: List[Dict[str, Any]] = field(default_factory=list)
    
    async def on_new_turn(self, role: str, content: str, is_new_goal: bool = False):
        """
        Process a new turn. If is_new_goal, compress previous goal first.
        """
        if is_new_goal and self.current_goal_turns:
            # Compress the previous goal
            await self._compress_current_goal()
            self.current_goal_id += 1
        
        self.current_goal_turns.append(Turn(
            role=role, 
            content=content,
            goal_id=self.current_goal_id
        ))
    
    async def _compress_current_goal(self):
        """Compress current goal turns into a summary."""
        if not self.current_goal_turns:
            return
        
        # Count tokens before compression
        original_text = self._format_turns(self.current_goal_turns)
        original_tokens = len(original_text) // 4
        
        # Generate summary
        if self.summarizer:
            summary = await self.summarizer.summarize(self.current_goal_turns)
        else:
            # Fallback: just use first and last turn
            summary = f"Goal {self.current_goal_id + 1}: {self.current_goal_turns[0].content[:100]}..."
        
        # Track compression event
        self._compression_events.append({
            "goal_id": self.current_goal_id,
            "turns_compressed": len(self.current_goal_turns),
            "original_tokens": original_tokens,
            "summary_tokens": len(summary) // 4,
            "summary": summary
        })
        
        self.goal_summaries.append(summary)
        self.current_goal_turns = []
        
        logger.info(f"Compressed goal {self.current_goal_id}: {self._compression_events[-1]['turns_compressed']} turns -> summary")
    
    def _format_turns(self, turns: List[Turn]) -> str:
        """Format turns into a string."""
        lines = []
        for turn in turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)
    
    def get_context(self) -> str:
        """Return summaries of completed goals + full current goal turns."""
        parts = []
        
        # Add summaries of completed goals
        if self.goal_summaries:
            summaries_text = "\n".join([
                f"[Previous Goal {i+1}: {s}]" 
                for i, s in enumerate(self.goal_summaries)
            ])
            parts.append(summaries_text)
        
        # Add current goal turns in full
        if self.current_goal_turns:
            current_text = self._format_turns(self.current_goal_turns)
            parts.append(current_text)
        
        return "\n\n".join(parts)
    
    def reset(self):
        """Reset for a new conversation."""
        self.goal_summaries = []
        self.current_goal_turns = []
        self.current_goal_id = 0
        self._compression_events = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        total_original = sum(e["original_tokens"] for e in self._compression_events)
        total_summary = sum(e["summary_tokens"] for e in self._compression_events)
        
        return {
            "strategy": "goal_boundary",
            "goals_compressed": len(self.goal_summaries),
            "current_goal_turns": len(self.current_goal_turns),
            "compression_events": self._compression_events,
            "total_tokens_before_compression": total_original,
            "total_tokens_after_compression": total_summary,
            "compression_ratio": total_summary / total_original if total_original > 0 else 1.0
        }


@dataclass 
class TokenBudgetStrategy(ContextStrategy):
    """
    Maintain a token budget. Compress oldest goals when budget exceeded.
    Combines token awareness with goal-boundary compression.
    """
    
    max_tokens: int = 4000
    summarizer: Optional['GoalSummarizer'] = None
    
    # Internal state - track goals separately
    goals: List[Dict[str, Any]] = field(default_factory=list)  # Each has turns, summary, is_summarized
    current_goal_turns: List[Turn] = field(default_factory=list)
    _total_compressions: int = field(default=0)
    
    def __post_init__(self):
        """Initialize with empty first goal."""
        self.goals = []
    
    async def on_new_turn(self, role: str, content: str, is_new_goal: bool = False):
        """Process turn, starting new goal if indicated."""
        if is_new_goal and self.current_goal_turns:
            # Save current goal
            self.goals.append({
                "turns": self.current_goal_turns.copy(),
                "summary": None,
                "is_summarized": False
            })
            self.current_goal_turns = []
        
        self.current_goal_turns.append(Turn(role=role, content=content))
        
        # Check if we need to compress
        await self._enforce_budget()
    
    async def _enforce_budget(self):
        """Compress oldest goals if over budget."""
        context = self.get_context()
        current_tokens = len(context) // 4
        
        while current_tokens > self.max_tokens:
            # Find oldest non-summarized goal
            compressed = False
            for goal in self.goals:
                if not goal["is_summarized"]:
                    await self._compress_goal(goal)
                    compressed = True
                    break
            
            if not compressed:
                # All goals already compressed, can't reduce further
                break
            
            context = self.get_context()
            current_tokens = len(context) // 4
    
    async def _compress_goal(self, goal: Dict[str, Any]):
        """Compress a single goal."""
        if self.summarizer:
            goal["summary"] = await self.summarizer.summarize(goal["turns"])
        else:
            goal["summary"] = f"[Goal: {goal['turns'][0].content[:50]}...]"
        goal["is_summarized"] = True
        self._total_compressions += 1
    
    def get_context(self) -> str:
        """Build context respecting summaries."""
        parts = []
        
        for i, goal in enumerate(self.goals):
            if goal["is_summarized"]:
                parts.append(f"[Previous Goal {i+1}: {goal['summary']}]")
            else:
                turns_text = "\n".join([
                    f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
                    for t in goal["turns"]
                ])
                parts.append(turns_text)
        
        # Add current goal
        if self.current_goal_turns:
            current_text = "\n".join([
                f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
                for t in self.current_goal_turns
            ])
            parts.append(current_text)
        
        return "\n\n".join(parts)
    
    def reset(self):
        """Reset for new conversation."""
        self.goals = []
        self.current_goal_turns = []
        self._total_compressions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "strategy": "token_budget",
            "max_tokens": self.max_tokens,
            "total_goals": len(self.goals) + (1 if self.current_goal_turns else 0),
            "goals_summarized": sum(1 for g in self.goals if g["is_summarized"]),
            "total_compressions": self._total_compressions,
            "current_context_tokens": len(self.get_context()) // 4
        }
