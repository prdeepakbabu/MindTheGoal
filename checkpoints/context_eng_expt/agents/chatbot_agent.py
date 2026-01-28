"""Chatbot agent with pluggable context management strategy and tool support."""

import logging
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')
from agents.bedrock_client import BedrockClient

if TYPE_CHECKING:
    from context_eng_expt.context.strategies import ContextStrategy
    from context_eng_expt.tools.router import ToolRouter

logger = logging.getLogger(__name__)


CHATBOT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. You can help users with various tasks.

## Available Tools

{tools_description}

## Tool Usage Format

When you need to use a tool, respond with a tool call in this format:
<tool_use>
<name>tool_name</name>
<arguments>{{"param1": "value1", "param2": "value2"}}</arguments>
</tool_use>

After a tool returns results, incorporate them into your response to the user.

## Guidelines

- Use tools when appropriate to provide accurate, up-to-date information
- For web searches, formulate clear, specific queries
- For code execution, use it for calculations or data processing
- Always explain what you're doing when using tools
- If a tool fails, try an alternative approach or explain the limitation
- Be helpful, concise, and proactive

Respond naturally and helpfully to the user's requests."""


CHATBOT_SYSTEM_PROMPT_NO_TOOLS = """You are a helpful travel and booking assistant. You can help users with:
- Restaurant reservations (find restaurants, check availability, make bookings)
- Hotel bookings (find hotels, check rooms, make reservations)  
- Taxi/transportation arrangements (book taxis, find routes)
- Attraction recommendations and ticket bookings

Be helpful, concise, and proactive. When you have enough information to complete a booking, 
confirm the details with the user before finalizing. If you need more information, ask 
clarifying questions.

Respond naturally and helpfully to the user's requests."""


@dataclass
class ToolUseMetrics:
    """Metrics for tool usage."""
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    tools_used: Dict[str, int] = field(default_factory=dict)
    
    def record(self, tool_name: str, success: bool):
        """Record a tool use."""
        self.total_tool_calls += 1
        if success:
            self.successful_tool_calls += 1
        else:
            self.failed_tool_calls += 1
        self.tools_used[tool_name] = self.tools_used.get(tool_name, 0) + 1
    
    @property
    def accuracy(self) -> float:
        """Tool use accuracy."""
        if self.total_tool_calls == 0:
            return 1.0
        return self.successful_tool_calls / self.total_tool_calls


@dataclass
class ChatbotAgent:
    """
    Chatbot agent that responds to user messages using a configurable context strategy.
    
    The context strategy determines how conversation history is managed and compressed.
    Supports optional tool use for enhanced capabilities.
    """
    
    context_strategy: Optional['ContextStrategy'] = None
    tool_router: Optional['ToolRouter'] = None
    system_prompt: Optional[str] = None
    max_tool_iterations: int = 3
    
    # Internal state
    _llm: Optional[BedrockClient] = field(default=None, init=False)
    _raw_history: List[Dict[str, str]] = field(default_factory=list, init=False)
    _token_counts: List[int] = field(default_factory=list, init=False)
    _tool_metrics: ToolUseMetrics = field(default_factory=ToolUseMetrics, init=False)
    
    def __post_init__(self):
        """Initialize the LLM client."""
        self._llm = BedrockClient()
        
        # Build system prompt
        if self.system_prompt is None:
            if self.tool_router:
                tools_desc = self.tool_router.get_tools_description()
                self.system_prompt = CHATBOT_SYSTEM_PROMPT.format(tools_description=tools_desc)
            else:
                self.system_prompt = CHATBOT_SYSTEM_PROMPT_NO_TOOLS
    
    async def respond(self, user_message: str, is_new_goal: bool = False) -> str:
        """
        Generate a response to the user's message.
        
        Args:
            user_message: The user's message
            is_new_goal: Whether this message starts a new goal (triggers compression)
            
        Returns:
            The chatbot's response
        """
        # Store raw turn
        self._raw_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Notify context strategy of new turn
        if self.context_strategy:
            await self.context_strategy.on_new_turn(
                role="user",
                content=user_message,
                is_new_goal=is_new_goal
            )
        
        # Build context for LLM
        context = self._build_context()
        
        # Count tokens for metrics
        input_tokens = self._estimate_tokens(context)
        
        # Generate response (with tool loop if tools are available)
        try:
            if self.tool_router:
                response = await self._respond_with_tools(context)
            else:
                response = await self._llm.invoke(
                    prompt=context,
                    temperature=0.7,
                    system_prompt=self.system_prompt
                )
            
            # Store response
            self._raw_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Notify context strategy
            if self.context_strategy:
                await self.context_strategy.on_new_turn(
                    role="assistant",
                    content=response,
                    is_new_goal=False
                )
            
            # Track token usage
            self._token_counts.append(input_tokens)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating chatbot response: {e}")
            raise
    
    async def _respond_with_tools(self, context: str) -> str:
        """
        Generate response with tool use loop.
        
        Iteratively processes tool calls until a final response is generated.
        """
        current_context = context
        tool_results = []
        
        for iteration in range(self.max_tool_iterations):
            # Generate LLM response
            response = await self._llm.invoke(
                prompt=current_context,
                temperature=0.7,
                system_prompt=self.system_prompt
            )
            
            # Check for tool calls
            tool_calls = self.tool_router.parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls - return final response
                if tool_results:
                    # Clean up any tool_use tags from response
                    response = self._clean_tool_tags(response)
                return response
            
            # Execute tool calls
            for tool_call in tool_calls:
                logger.info(f"Executing tool: {tool_call.tool_name}")
                result = await self.tool_router.execute_tool_call(tool_call)
                
                # Record metrics
                self._tool_metrics.record(tool_call.tool_name, result.success)
                
                tool_results.append({
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "result": result.to_string()
                })
            
            # Build updated context with tool results
            tool_results_str = self._format_tool_results(tool_results[-len(tool_calls):])
            current_context = f"{current_context}\n\nAssistant: {response}\n\nTool Results:\n{tool_results_str}\n\nContinue your response incorporating the tool results:"
        
        # Max iterations reached - return last response
        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
        return self._clean_tool_tags(response)
    
    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool results for context."""
        lines = []
        for r in results:
            lines.append(f"Tool: {r['tool']}")
            lines.append(f"Result: {r['result']}")
            lines.append("")
        return "\n".join(lines)
    
    def _clean_tool_tags(self, text: str) -> str:
        """Remove tool_use tags from response."""
        # Remove <tool_use>...</tool_use> blocks
        cleaned = re.sub(r'<tool_use>.*?</tool_use>', '', text, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    def _build_context(self) -> str:
        """Build the context string for the LLM."""
        if self.context_strategy:
            return self.context_strategy.get_context()
        else:
            # Default: use full history
            return self._format_full_history()
    
    def _format_full_history(self) -> str:
        """Format the full conversation history."""
        lines = []
        for turn in self._raw_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about token usage and tool use."""
        full_context_tokens = self._estimate_tokens(self._format_full_history())
        
        if self.context_strategy:
            compressed_tokens = self._estimate_tokens(self.context_strategy.get_context())
        else:
            compressed_tokens = full_context_tokens
        
        metrics = {
            "total_turns": len(self._raw_history),
            "full_context_tokens": full_context_tokens,
            "compressed_context_tokens": compressed_tokens,
            "token_reduction_pct": (
                (full_context_tokens - compressed_tokens) / full_context_tokens * 100
                if full_context_tokens > 0 else 0
            ),
            "token_counts_per_turn": self._token_counts,
            "avg_input_tokens": (
                sum(self._token_counts) / len(self._token_counts)
                if self._token_counts else 0
            )
        }
        
        # Add tool metrics if tools are enabled
        if self.tool_router:
            metrics["tool_metrics"] = {
                "total_calls": self._tool_metrics.total_tool_calls,
                "successful_calls": self._tool_metrics.successful_tool_calls,
                "failed_calls": self._tool_metrics.failed_tool_calls,
                "accuracy": self._tool_metrics.accuracy,
                "tools_used": self._tool_metrics.tools_used
            }
        
        return metrics
    
    def get_raw_history(self) -> List[Dict[str, str]]:
        """Get the raw, uncompressed conversation history."""
        return self._raw_history.copy()
    
    def reset(self):
        """Reset the agent for a new conversation."""
        self._raw_history = []
        self._token_counts = []
        self._tool_metrics = ToolUseMetrics()
        if self.context_strategy:
            self.context_strategy.reset()
