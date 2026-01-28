"""Web search tool using DuckDuckGo."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import BaseTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    
    def __str__(self) -> str:
        return f"• {self.title}\n  {self.snippet}\n  URL: {self.url}"


class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo.
    
    Uses the duckduckgo-search library for free web searches.
    """
    
    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. Returns top search results with "
        "titles, snippets, and URLs. Use for finding information, looking up "
        "facts, or researching topics."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query to execute",
            required=True
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False,
            default=5
        )
    ]
    
    def __init__(self, timeout: int = 10):
        super().__init__()
        self.timeout = timeout
        self._ddg = None
    
    def _get_ddg(self):
        """Lazy load DuckDuckGo search client."""
        if self._ddg is None:
            try:
                from duckduckgo_search import DDGS
                self._ddg = DDGS()
            except ImportError:
                raise ImportError(
                    "duckduckgo-search not installed. "
                    "Run: pip install duckduckgo-search"
                )
        return self._ddg
    
    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Execute a web search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default: 5)
            
        Returns:
            ToolResult with list of SearchResult objects
        """
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output=None,
                error="Search query cannot be empty"
            )
        
        logger.info(f"Searching: '{query}' (max {max_results} results)")
        
        try:
            # Run synchronous DuckDuckGo search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._search_sync(query, max_results)
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    output=[],
                    metadata={"query": query, "result_count": 0}
                )
            
            search_results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", ""))
                )
                for r in results[:max_results]
            ]
            
            logger.info(f"Found {len(search_results)} results for '{query}'")
            
            return ToolResult(
                success=True,
                output=search_results,
                metadata={
                    "query": query,
                    "result_count": len(search_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Search failed: {str(e)}"
            )
    
    def _search_sync(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Synchronous search implementation."""
        ddg = self._get_ddg()
        results = list(ddg.text(query, max_results=max_results))
        return results
