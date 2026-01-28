"""Sandboxed Python code execution tool."""

import logging
import asyncio
import sys
import io
import traceback
from typing import Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr

from .base import BaseTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)

# Safe builtins for sandboxed execution
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
    'enumerate', 'filter', 'float', 'format', 'frozenset', 'hash',
    'hex', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
    'map', 'max', 'min', 'oct', 'ord', 'pow', 'print', 'range',
    'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
    'sum', 'tuple', 'type', 'zip'
}

# Safe modules that can be imported
SAFE_MODULES = {
    'math', 'statistics', 'random', 'datetime', 'json', 're',
    'collections', 'itertools', 'functools', 'decimal', 'fractions'
}


class CodeExecutorTool(BaseTool):
    """
    Sandboxed Python code execution tool.
    
    Executes Python code in a restricted environment for safety.
    Useful for calculations, data processing, and simple scripts.
    """
    
    name = "execute_python"
    description = (
        "Execute Python code for calculations, data processing, or analysis. "
        "Returns the output of the code. Limited to safe operations - no file "
        "system access or network calls. Can use: math, statistics, random, "
        "datetime, json, re, collections, itertools."
    )
    parameters = [
        ToolParameter(
            name="code",
            type="string",
            description="Python code to execute",
            required=True
        )
    ]
    
    def __init__(self, timeout: int = 10, max_output_length: int = 5000):
        super().__init__()
        self.timeout = timeout
        self.max_output_length = max_output_length
    
    async def execute(self, code: str) -> ToolResult:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            ToolResult with execution output or error
        """
        if not code or not code.strip():
            return ToolResult(
                success=False,
                output=None,
                error="Code cannot be empty"
            )
        
        logger.info(f"Executing code ({len(code)} chars)")
        
        try:
            # Run in thread pool with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._execute_sync(code)
                ),
                timeout=self.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error("Code execution timed out")
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution failed: {str(e)}"
            )
    
    def _execute_sync(self, code: str) -> ToolResult:
        """Synchronous code execution in sandbox."""
        # Create restricted globals
        restricted_globals = self._create_sandbox()
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile the code first to check for syntax errors
                compiled = compile(code, '<sandbox>', 'exec')
                
                # Execute in sandbox
                exec(compiled, restricted_globals)
            
            # Get output
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Check for result variable (last expression value)
            result = restricted_globals.get('result', None)
            
            # Combine output
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if result is not None:
                output_parts.append(f"Result: {result}")
            if stderr:
                output_parts.append(f"Warnings: {stderr}")
            
            output = "\n".join(output_parts) if output_parts else "Code executed successfully (no output)"
            
            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "... (truncated)"
            
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "code_length": len(code),
                    "output_length": len(output)
                }
            )
            
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Syntax error: {e}"
            )
        except NameError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Name error (possibly blocked import): {e}"
            )
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                output=None,
                error=f"Runtime error: {e}\n{tb}"
            )
    
    def _create_sandbox(self) -> Dict[str, Any]:
        """Create a restricted execution environment."""
        # Start with safe builtins
        safe_builtins = {
            name: getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, name, None)
            for name in SAFE_BUILTINS
            if hasattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, name)
        }
        
        # Handle __builtins__ being either a dict or module
        if isinstance(__builtins__, dict):
            safe_builtins = {name: __builtins__[name] for name in SAFE_BUILTINS if name in __builtins__}
        else:
            safe_builtins = {name: getattr(__builtins__, name) for name in SAFE_BUILTINS if hasattr(__builtins__, name)}
        
        # Add restricted __import__
        safe_builtins['__import__'] = self._restricted_import
        
        return {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }
    
    def _restricted_import(self, name: str, *args, **kwargs):
        """Only allow importing safe modules."""
        if name not in SAFE_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed. Safe modules: {SAFE_MODULES}")
        return __import__(name, *args, **kwargs)
