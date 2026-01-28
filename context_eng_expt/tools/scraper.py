"""Web scraper tool using BeautifulSoup."""

import logging
import asyncio
import re
from typing import Optional
from urllib.parse import urlparse

from .base import BaseTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class WebScraperTool(BaseTool):
    """
    Web scraper tool for extracting content from webpages.
    
    Uses requests + BeautifulSoup for HTML parsing.
    """
    
    name = "scrape_webpage"
    description = (
        "Extract text content from a webpage URL. Returns the main text content "
        "of the page. Useful for reading articles, documentation, or any web content."
    )
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="The URL of the webpage to scrape",
            required=True
        ),
        ToolParameter(
            name="selector",
            type="string",
            description="CSS selector to extract specific content (optional)",
            required=False,
            default=None
        ),
        ToolParameter(
            name="max_length",
            type="integer",
            description="Maximum length of content to return (default: 5000)",
            required=False,
            default=5000
        )
    ]
    
    def __init__(self, timeout: int = 15, max_content_length: int = 10000):
        super().__init__()
        self.timeout = timeout
        self.max_content_length = max_content_length
        self._session = None
    
    def _validate_url(self, url: str) -> bool:
        """Validate that the URL is safe to scrape."""
        try:
            parsed = urlparse(url)
            # Only allow http/https
            if parsed.scheme not in ("http", "https"):
                return False
            # Must have a host
            if not parsed.netloc:
                return False
            return True
        except Exception:
            return False
    
    async def execute(
        self, 
        url: str, 
        selector: Optional[str] = None,
        max_length: int = 5000
    ) -> ToolResult:
        """
        Scrape content from a webpage.
        
        Args:
            url: URL to scrape
            selector: Optional CSS selector to target specific content
            max_length: Maximum characters to return
            
        Returns:
            ToolResult with extracted text content
        """
        if not self._validate_url(url):
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid URL: {url}"
            )
        
        logger.info(f"Scraping: {url}")
        
        try:
            # Run synchronous requests in thread pool
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                lambda: self._scrape_sync(url, selector, max_length)
            )
            
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "url": url,
                    "selector": selector,
                    "content_length": len(content)
                }
            )
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Scraping failed: {str(e)}"
            )
    
    def _scrape_sync(
        self, 
        url: str, 
        selector: Optional[str],
        max_length: int
    ) -> str:
        """Synchronous scraping implementation."""
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch the page
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        
        response = requests.get(
            url, 
            headers=headers, 
            timeout=self.timeout,
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Extract content
        if selector:
            elements = soup.select(selector)
            if elements:
                text = "\n".join(el.get_text(strip=True) for el in elements)
            else:
                text = f"No elements found matching selector: {selector}"
        else:
            # Try to find main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
