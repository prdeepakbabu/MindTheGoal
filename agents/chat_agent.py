"""Chat Agent for generating AI assistant responses."""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from agents.bedrock_client import BedrockClient
from config import get_settings

logger = logging.getLogger(__name__)


CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant engaged in a task-oriented conversation. 

Your role is to:
1. Understand the user's goals and requests
2. Provide helpful, accurate, and relevant responses
3. Ask clarifying questions when needed
4. Guide users through multi-step tasks

Be concise but thorough. If you don't know something or can't help with a request, say so clearly.

Current conversation context will be provided. Respond naturally as part of an ongoing dialogue."""


class ChatAgent:
    """
    AI Assistant agent for generating conversation responses.
    
    Used for demonstrating the evaluation system in real-time
    or for generating synthetic conversations.
    """
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize Chat Agent.
        
        Args:
            bedrock_client: Optional pre-configured Bedrock client
            system_prompt: Custom system prompt
            temperature: Sampling temperature (higher for creativity)
        """
        settings = get_settings()
        
        self.temperature = temperature or settings.chat_temperature
        self.system_prompt = system_prompt or CHAT_SYSTEM_PROMPT
        
        if bedrock_client:
            self.client = bedrock_client
        else:
            self.client = BedrockClient(temperature=self.temperature)
        
        self.conversation_history: List[Dict[str, str]] = []
    
    async def respond(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response to a user message.
        
        Args:
            user_message: The user's message
            conversation_history: Optional conversation history
            
        Returns:
            Agent's response text
        """
        history = conversation_history or self.conversation_history
        
        # Build prompt with conversation context
        prompt = self._build_prompt(user_message, history)
        
        response = await self.client.invoke(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature
        )
        
        return response.strip()
    
    async def respond_streaming(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.
        
        Args:
            user_message: The user's message
            conversation_history: Optional conversation history
            
        Yields:
            Response text chunks
        """
        history = conversation_history or self.conversation_history
        prompt = self._build_prompt(user_message, history)
        
        async for chunk in self.client.invoke_streaming(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature
        ):
            yield chunk
    
    def _build_prompt(
        self,
        user_message: str,
        history: List[Dict[str, str]]
    ) -> str:
        """Build prompt with conversation history."""
        lines = []
        
        if history:
            lines.append("Previous conversation:")
            for msg in history:
                lines.append(f"User: {msg.get('user', '')}")
                lines.append(f"Assistant: {msg.get('agent', '')}")
            lines.append("")
        
        lines.append(f"User: {user_message}")
        lines.append("Assistant:")
        
        return "\n".join(lines)
    
    def add_to_history(self, user_message: str, agent_response: str) -> None:
        """Add a turn to conversation history."""
        self.conversation_history.append({
            "user": user_message,
            "agent": agent_response
        })
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()


class TaskOrientedChatAgent(ChatAgent):
    """
    Specialized chat agent for task-oriented dialogues.
    
    Simulates a task-completion assistant with specific domains.
    """
    
    DOMAIN_PROMPTS = {
        "restaurant": """You are a restaurant booking assistant. Help users:
- Find restaurants by cuisine, location, price range
- Make reservations
- Get restaurant information (hours, menu, etc.)
Available information: Restaurant name, cuisine type, price range, area, phone number, address.""",

        "hotel": """You are a hotel booking assistant. Help users:
- Find hotels by location, star rating, price
- Make room reservations
- Get hotel amenities and information
Available information: Hotel name, star rating, price range, area, type, parking, internet.""",

        "train": """You are a train booking assistant. Help users:
- Find train schedules between destinations
- Book train tickets
- Get travel time and pricing information
Available information: Departure, destination, day, leave time, arrive time, duration, price.""",

        "taxi": """You are a taxi booking assistant. Help users:
- Book taxis for pickup/dropoff
- Get fare estimates
- Arrange specific pickup times
Available information: Pickup location, destination, car type, phone number.""",

        "attraction": """You are a tourism information assistant. Help users:
- Find tourist attractions
- Get opening hours and entrance fees
- Provide location and contact information
Available information: Attraction name, type, area, address, phone, entrance fee.""",
    }
    
    def __init__(
        self,
        domain: str = "general",
        **kwargs
    ):
        """
        Initialize task-oriented chat agent.
        
        Args:
            domain: Task domain (restaurant, hotel, train, taxi, attraction)
            **kwargs: Additional arguments for ChatAgent
        """
        system_prompt = self.DOMAIN_PROMPTS.get(domain, CHAT_SYSTEM_PROMPT)
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.domain = domain
