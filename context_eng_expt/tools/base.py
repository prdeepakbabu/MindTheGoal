"""Base classes for tools."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Exception raised when a tool execution fails."""
    pass


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert result to string for LLM context."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            elif isinstance(self.output, list):
                return "\n".join(str(item) for item in self.output[:10])
            else:
                return str(self.output)
        else:
            return f"Error: {self.error}"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class BaseTool(ABC):
    """
    Base class for all tools.
    
    Tools are callable capabilities that the agent can use to interact
    with the outside world (search, scrape, execute code, etc.).
    """
    
    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: List[ToolParameter] = []
    
    def __init__(self):
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    async def __call__(self, **kwargs) -> ToolResult:
        """Call the tool, tracking metrics."""
        self._call_count += 1
        try:
            result = await self.execute(**kwargs)
            if result.success:
                self._success_count += 1
            else:
                self._error_count += 1
            return result
        except Exception as e:
            self._error_count += 1
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": (
                self._success_count / self._call_count 
                if self._call_count > 0 else 0.0
            )
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0
