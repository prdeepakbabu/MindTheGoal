"""AWS Bedrock client wrapper for LLM interactions."""

import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, TokenRetrievalError, CredentialRetrievalError

from config import get_settings

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    AWS Bedrock client for Claude model interactions.
    
    Handles authentication, request formatting, and response parsing
    for Claude 3.5 Sonnet via AWS Bedrock.
    """
    
    # Token/credential expiry related exceptions
    TOKEN_EXPIRY_ERRORS = (
        'ExpiredToken',
        'ExpiredTokenException',
        'TokenRefreshRequired',
        'InvalidIdentityToken',
        'UnauthorizedException',
        'AccessDeniedException'
    )
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize Bedrock client.
        
        Args:
            model_id: Bedrock model ID (defaults to config)
            region: AWS region (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        settings = get_settings()
        
        self.model_id = model_id or settings.bedrock_model_id
        self.region = region or settings.aws_region
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._create_client()
        
        logger.info(f"Initialized Bedrock client for {self.model_id} in {self.region}")
    
    def _create_client(self):
        """Create a fresh boto3 client with new credentials."""
        # Clear any cached credentials by creating a new session
        session = boto3.Session()
        
        config = Config(
            region_name=self.region,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
        
        self._client = session.client(
            "bedrock-runtime",
            config=config
        )
        
        logger.info("Created new Bedrock client with fresh credentials")
    
    def _is_token_expiry_error(self, error: Exception) -> bool:
        """Check if the error is related to token/credential expiry."""
        if isinstance(error, (TokenRetrievalError, CredentialRetrievalError)):
            return True
        
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            if error_code in self.TOKEN_EXPIRY_ERRORS:
                return True
            error_message = str(error).lower()
            if any(x.lower() in error_message for x in ['expired', 'token', 'credential', 'unauthorized']):
                return True
        
        # Check string representation for common patterns
        error_str = str(error).lower()
        return any(x.lower() in error_str for x in ['expiredtoken', 'token expired', 'credential', 'unauthorized'])
    
    def _refresh_and_retry(self):
        """Refresh credentials by recreating the client."""
        logger.warning("Token/credential expiry detected, refreshing client...")
        self._create_client()
    
    async def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Invoke the model with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Model response text
        """
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response["body"].read())
                
                # Extract text from Claude response format
                if "content" in response_body:
                    return response_body["content"][0]["text"]
                
                return response_body.get("completion", "")
                
            except Exception as e:
                last_error = e
                if self._is_token_expiry_error(e) and attempt < max_retries - 1:
                    logger.warning(f"Token expiry detected on attempt {attempt + 1}, refreshing credentials...")
                    self._refresh_and_retry()
                    continue
                logger.error(f"Bedrock invocation failed: {e}")
                raise
        
        raise last_error
    
    async def invoke_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Invoke the model with streaming response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Yields:
            Response text chunks
        """
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                
                if chunk["type"] == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
                        
        except Exception as e:
            logger.error(f"Bedrock streaming invocation failed: {e}")
            raise
    
    async def invoke_with_json_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Invoke and parse JSON response.
        
        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            temperature: Override temperature
            
        Returns:
            Parsed JSON dictionary
        """
        response = await self.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Try to extract JSON from response
        try:
            # Look for JSON in code blocks
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return {"raw_response": response, "parse_error": str(e)}
