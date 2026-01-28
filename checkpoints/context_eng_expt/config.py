"""Configuration for the context compaction experiment."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class CompactionStrategy(Enum):
    """Available context compaction strategies."""
    FULL_CONTEXT = "full"
    SLIDING_WINDOW = "sliding_window"
    GOAL_BOUNDARY = "goal_boundary"
    TOKEN_BUDGET = "token_budget"


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    
    # Strategy settings
    strategy: CompactionStrategy = CompactionStrategy.GOAL_BOUNDARY
    sliding_window_size: int = 10
    token_budget: int = 4000
    
    # Experiment settings
    max_turns_per_conversation: int = 25
    num_runs_per_scenario: int = 2
    
    # Output settings
    output_dir: str = "context_eng_expt/results"
    save_conversations: bool = True
    
    # Tool settings
    enable_tools: bool = True


# Scenarios that require tool use and span multiple goals
SCENARIOS = [
    {
        "id": "research_and_plan",
        "description": "Research a topic and create a plan",
        "goals": [
            "Search for information about renewable energy trends in 2024",
            "Calculate the cost savings of solar panels for a household"
        ],
        "expected_turns": 8,
        "tools_expected": ["web_search", "execute_python"]
    },
    {
        "id": "multi_topic_research",
        "description": "Research multiple related topics",
        "goals": [
            "Find current weather impacts on agriculture",
            "Search for drought-resistant crop varieties",
            "Calculate water savings with drip irrigation"
        ],
        "expected_turns": 12,
        "tools_expected": ["web_search", "execute_python"]
    },
    {
        "id": "fact_check_and_analyze",
        "description": "Fact check claims and do analysis",
        "goals": [
            "Search for data on global electric vehicle adoption",
            "Calculate the growth rate year over year",
            "Find information about charging infrastructure"
        ],
        "expected_turns": 12,
        "tools_expected": ["web_search", "execute_python"]
    },
    {
        "id": "comparison_research",
        "description": "Compare options and make calculations",
        "goals": [
            "Search for information about Python vs JavaScript for web development",
            "Find salary data for web developers",
            "Calculate learning time investment for each"
        ],
        "expected_turns": 10,
        "tools_expected": ["web_search", "execute_python"]
    }
]


# User personas
PERSONAS = [
    {
        "id": "efficient",
        "description": "Direct user who knows what they want",
        "prompt_modifier": "Be direct and efficient. State your requirements clearly. Ask for specific facts and numbers."
    },
    {
        "id": "curious",
        "description": "Asks follow-up questions, wants to understand deeply",
        "prompt_modifier": "Ask follow-up questions. Request more details. Want to understand the reasoning behind information."
    },
    {
        "id": "analytical",
        "description": "Wants data, calculations, and comparisons",
        "prompt_modifier": "Focus on numbers and data. Ask for calculations. Request comparisons and statistics."
    }
]


# System prompts for different components
SIMULATED_USER_SYSTEM_PROMPT = """You are simulating a user interacting with an AI assistant.

## Your Character
Persona: {persona}
{prompt_modifier}

## Your Goals (complete in order)
{goals}

## Current State
Current goal: {current_goal}
Goals completed: {completed_goals}

## Instructions
1. Generate the next user message to work toward your current goal
2. Ask questions that require the assistant to:
   - Search for real information (web search)
   - Perform calculations (math, statistics)
   - Analyze data or compare options
3. When the assistant has adequately addressed your current goal, acknowledge it and move to the next goal
4. Be natural - ask clarifying questions, request specific data, follow up on interesting points
5. When all goals are complete, end the conversation naturally

## Important
- Do NOT be overly cooperative - require the assistant to actually help you
- Ask for SPECIFIC facts, numbers, or calculations
- Reference information from previous goals when relevant (tests context retention)

Generate ONLY the user's next message, nothing else."""


GOAL_DETECTOR_PROMPT = """Analyze this user message to determine if it represents a NEW goal or continues the current goal.

Current conversation context:
{context}

New user message:
{user_message}

Previous agent response (if any):
{agent_response}

A NEW GOAL is indicated when the user:
- Explicitly changes topic ("Now I want to..." "Moving on..." "Next question...")
- Asks about something unrelated to the current discussion
- Thanks for help with current topic and asks something new
- Starts a clearly different task

The SAME GOAL continues when the user:
- Asks follow-up questions about the current topic
- Requests more details or clarification
- Provides additional constraints or preferences
- Continues the same line of inquiry

Respond with ONLY one word: NEW or CONTINUE"""


CHATBOT_SYSTEM_PROMPT_WITH_TOOLS = """You are a helpful AI assistant with access to tools.

## Available Tools

{tools_description}

## Tool Usage Format

When you need to use a tool, respond with:
<tool_use>
<name>tool_name</name>
<arguments>{{"param1": "value1"}}</arguments>
</tool_use>

## Guidelines

1. Use web_search for current information, facts, statistics
2. Use execute_python for calculations, math, data analysis
3. Use scrape_webpage only if you need detailed content from a specific URL
4. Always explain what you're doing and why
5. After getting tool results, synthesize and present them clearly
6. If a tool fails, explain and try alternatives

Be helpful, accurate, and thorough."""
