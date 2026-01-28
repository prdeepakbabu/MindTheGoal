"""Tool router for routing LLM tool calls to appropriate tools."""

import logging
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .base import BaseTool, ToolResult
from .search import WebSearchTool
from .scraper import WebScraperTool
from .code_executor import CodeExecutorTool

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output."""
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str = ""


class ToolRouter:
    """
    Routes tool calls from LLM output to the appropriate tool implementations.
    
    Handles parsing tool calls from various formats and executing them.
    """
    
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        """
        Initialize the router with available tools.
        
        Args:
            tools: List of tools to register. If None, registers default tools.
        """
        self.tools: Dict[str, BaseTool] = {}
        
        if tools is None:
            # Register default tools
            self._register_defaults()
        else:
            for tool in tools:
                self.register(tool)
    
    def _register_defaults(self):
        """Register the default set of tools."""
        self.register(WebSearchTool())
        self.register(WebScraperTool())
        self.register(CodeExecutorTool())
    
    def register(self, tool: BaseTool):
        """Register a tool with the router."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools (for LLM function calling)."""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all tools for the system prompt."""
        lines = ["Available tools:"]
        for tool in self.tools.values():
            params = ", ".join([
                f"{p.name}: {p.type}" + ("" if p.required else " (optional)")
                for p in tool.parameters
            ])
            lines.append(f"\n• {tool.name}({params})")
            lines.append(f"  {tool.description}")
        return "\n".join(lines)
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            ToolResult from the tool execution
        """
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            )
        
        tool = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
        
        return await tool(**arguments)
    
    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a parsed ToolCall object."""
        return await self.execute(tool_call.tool_name, tool_call.arguments)
    
    def parse_tool_calls(self, llm_output: str) -> List[ToolCall]:
        """
        Parse tool calls from LLM output.
        
        Supports multiple formats:
        1. JSON format: {"tool": "name", "arguments": {...}}
        2. XML-like format: <tool_use><name>tool_name</name><arguments>...</arguments></tool_use>
        3. Function-like format: tool_name(arg1="value1", arg2="value2")
        
        Args:
            llm_output: Raw LLM output text
            
        Returns:
            List of parsed ToolCall objects
        """
        tool_calls = []
        
        # Try JSON format
        json_calls = self._parse_json_tool_calls(llm_output)
        tool_calls.extend(json_calls)
        
        # Try XML format
        xml_calls = self._parse_xml_tool_calls(llm_output)
        tool_calls.extend(xml_calls)
        
        # Try function format
        func_calls = self._parse_function_tool_calls(llm_output)
        tool_calls.extend(func_calls)
        
        return tool_calls
    
    def _parse_json_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse JSON-formatted tool calls."""
        calls = []
        
        # Look for JSON objects with tool/name and arguments
        json_pattern = r'\{[^{}]*"(?:tool|name)"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                tool_name = data.get("tool") or data.get("name")
                arguments = data.get("arguments", data.get("args", {}))
                
                if tool_name and tool_name in self.tools:
                    calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments=arguments if isinstance(arguments, dict) else {},
                        raw_text=match
                    ))
            except json.JSONDecodeError:
                continue
        
        return calls
    
    def _parse_xml_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse XML-formatted tool calls."""
        calls = []
        
        # Look for <tool_use> blocks
        xml_pattern = r'<tool_use>(.*?)</tool_use>'
        matches = re.findall(xml_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # Extract tool name
                name_match = re.search(r'<name>(.*?)</name>', match)
                if not name_match:
                    continue
                tool_name = name_match.group(1).strip()
                
                # Extract arguments
                args_match = re.search(r'<arguments>(.*?)</arguments>', match, re.DOTALL)
                if args_match:
                    try:
                        arguments = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = {}
                
                if tool_name in self.tools:
                    calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        raw_text=f"<tool_use>{match}</tool_use>"
                    ))
            except Exception:
                continue
        
        return calls
    
    def _parse_function_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse function-style tool calls like tool_name(arg='value')."""
        calls = []
        
        # Build pattern from registered tool names
        tool_names = "|".join(re.escape(name) for name in self.tools.keys())
        if not tool_names:
            return calls
        
        func_pattern = rf'({tool_names})\s*\((.*?)\)'
        matches = re.findall(func_pattern, text, re.DOTALL)
        
        for tool_name, args_str in matches:
            try:
                # Parse arguments
                arguments = self._parse_function_args(args_str)
                calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_text=f"{tool_name}({args_str})"
                ))
            except Exception:
                continue
        
        return calls
    
    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """Parse function-style arguments."""
        arguments = {}
        if not args_str.strip():
            return arguments
        
        # Handle key=value pairs
        # This is a simple parser - handles basic cases
        arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))'
        matches = re.findall(arg_pattern, args_str)
        
        for key, val1, val2, val3 in matches:
            value = val1 or val2 or val3
            # Try to convert to appropriate type
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            arguments[key] = value
        
        return arguments
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all tools."""
        return {
            "tools": {
                name: tool.get_stats()
                for name, tool in self.tools.items()
            },
            "total_calls": sum(t._call_count for t in self.tools.values()),
            "total_successes": sum(t._success_count for t in self.tools.values()),
            "total_errors": sum(t._error_count for t in self.tools.values())
        }
    
    def reset_all_stats(self):
        """Reset statistics for all tools."""
        for tool in self.tools.values():
            tool.reset_stats()
