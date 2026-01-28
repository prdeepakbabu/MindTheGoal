"""Custom JSON dataset loader."""

import os
import json
import random
from typing import List, Optional, Dict, Any
import logging

from datasets.base_loader import BaseDatasetLoader
from core.models import Session, Turn

logger = logging.getLogger(__name__)


class CustomLoader(BaseDatasetLoader):
    """
    Loader for custom JSON format dialogues.
    
    Supports flexible JSON formats for your own conversation data.
    """
    
    name = "custom"
    description = "Custom JSON format dialogues"
    
    def __init__(self, data_dir: Optional[str] = None, file_path: Optional[str] = None):
        super().__init__(data_dir)
        self.file_path = file_path
    
    async def load(self) -> List[Session]:
        """Load custom dataset from JSON files."""
        logger.info("Loading custom dataset...")
        
        sessions = []
        
        if self.file_path and os.path.exists(self.file_path):
            sessions = await self._load_file(self.file_path)
        elif self.data_dir and os.path.exists(self.data_dir):
            sessions = await self._load_directory()
        else:
            raise RuntimeError("No valid file_path or data_dir provided")
        
        self._sessions = sessions
        self._loaded = True
        logger.info(f"Loaded {len(sessions)} sessions from custom dataset")
        return sessions
    
    async def _load_file(self, filepath: str) -> List[Session]:
        """Load sessions from a single JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return self._parse_data(data)
    
    async def _load_directory(self) -> List[Session]:
        """Load sessions from all JSON files in directory."""
        sessions = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                sessions.extend(self._parse_data(data))
        
        return sessions
    
    def _parse_data(self, data: Any) -> List[Session]:
        """Parse JSON data into sessions."""
        sessions = []
        
        if isinstance(data, list):
            for item in data:
                session = self._convert_dialogue(item)
                if session:
                    sessions.append(session)
        elif isinstance(data, dict):
            if "sessions" in data:
                for item in data["sessions"]:
                    session = self._convert_dialogue(item)
                    if session:
                        sessions.append(session)
            elif "dialogues" in data:
                for item in data["dialogues"]:
                    session = self._convert_dialogue(item)
                    if session:
                        sessions.append(session)
            else:
                session = self._convert_dialogue(data)
                if session:
                    sessions.append(session)
        
        return sessions
    
    def _convert_dialogue(self, item: Dict[str, Any]) -> Optional[Session]:
        """
        Convert flexible JSON format to Session.
        
        Supports multiple formats:
        - Format 1: {"turns": [{"user": "...", "agent": "..."}]}
        - Format 2: {"messages": [{"role": "user/assistant", "content": "..."}]}
        - Format 3: {"conversation": [["user msg", "agent msg"], ...]}
        """
        try:
            session_id = item.get("session_id") or item.get("id") or str(id(item))
            
            turns = []
            turn_number = 1
            
            # Format 1: turns array with user/agent keys
            if "turns" in item:
                for turn_data in item["turns"]:
                    user_msg = turn_data.get("user") or turn_data.get("user_message") or turn_data.get("input")
                    agent_msg = turn_data.get("agent") or turn_data.get("agent_response") or turn_data.get("output") or turn_data.get("assistant")
                    
                    if user_msg and agent_msg:
                        turns.append(Turn(
                            turn_number=turn_number,
                            user_message=user_msg,
                            agent_response=agent_msg
                        ))
                        turn_number += 1
            
            # Format 2: messages array with role/content
            elif "messages" in item:
                user_msg = None
                for msg in item["messages"]:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    
                    if role in ["user", "human"]:
                        user_msg = content
                    elif role in ["assistant", "agent", "bot", "system"] and user_msg:
                        turns.append(Turn(
                            turn_number=turn_number,
                            user_message=user_msg,
                            agent_response=content
                        ))
                        turn_number += 1
                        user_msg = None
            
            # Format 3: conversation array of pairs
            elif "conversation" in item:
                for pair in item["conversation"]:
                    if isinstance(pair, list) and len(pair) >= 2:
                        turns.append(Turn(
                            turn_number=turn_number,
                            user_message=pair[0],
                            agent_response=pair[1]
                        ))
                        turn_number += 1
            
            if not turns:
                return None
            
            return Session(
                session_id=session_id,
                turns=turns,
                metadata={
                    "source": "custom",
                    "domains": item.get("domains", []),
                    **item.get("metadata", {})
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert custom dialogue: {e}")
            return None
    
    async def load_sample(
        self,
        sample_size: int,
        random_seed: Optional[int] = None
    ) -> List[Session]:
        """Load a sample of dialogues."""
        if not self._loaded:
            await self.load()
        
        if sample_size >= len(self._sessions):
            return self._sessions
        
        if random_seed is not None:
            random.seed(random_seed)
        
        return random.sample(self._sessions, sample_size)
