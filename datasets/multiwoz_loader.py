"""MultiWOZ 2.4 dataset loader."""

import os
import json
import random
from typing import List, Optional, Dict, Any
import logging

from datasets.base_loader import BaseDatasetLoader
from core.models import Session, Turn

logger = logging.getLogger(__name__)


class MultiWOZLoader(BaseDatasetLoader):
    """
    Loader for MultiWOZ 2.4 dataset.
    
    MultiWOZ is a multi-domain task-oriented dialogue dataset with
    ~10,000 dialogues across 7 domains (Restaurant, Hotel, Attraction,
    Taxi, Train, Hospital, Police).
    """
    
    name = "multiwoz"
    description = "MultiWOZ 2.4 - Multi-domain task-oriented dialogues"
    
    # Hugging Face dataset name
    HF_DATASET = "pfb30/multi_woz_v22"
    
    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self._raw_data: Dict[str, Any] = {}
    
    async def load(self) -> List[Session]:
        """Load MultiWOZ dataset."""
        logger.info("Loading MultiWOZ dataset...")
        
        # Try local files first if available
        if self.data_dir and os.path.exists(self.data_dir):
            logger.info(f"Loading from local directory: {self.data_dir}")
            sessions = await self._load_from_local()
        else:
            # Try sample_dialogues.json as fallback
            sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_dialogues.json")
            if os.path.exists(sample_path):
                logger.info(f"Loading from sample dialogues: {sample_path}")
                sessions = await self._load_from_sample(sample_path)
            else:
                try:
                    # Try loading from Hugging Face datasets
                    sessions = await self._load_from_huggingface()
                except Exception as e:
                    logger.warning(f"Failed to load from HuggingFace: {e}")
                    raise RuntimeError(
                        "Could not load MultiWOZ. Install 'datasets' package or "
                        "provide local data directory."
                    )
        
        self._sessions = sessions
        self._loaded = True
        logger.info(f"Loaded {len(sessions)} sessions from MultiWOZ")
        return sessions
    
    async def _load_from_huggingface(self) -> List[Session]:
        """Load MultiWOZ from HuggingFace datasets."""
        import sys
        import subprocess
        
        # Run in a subprocess from temp dir to avoid module conflicts with our local datasets/
        import tempfile
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
import json
from datasets import load_dataset

# Use multi_woz_v22 from HuggingFace
ds = load_dataset("multi_woz_v22")
output = []
for split in ["train", "validation", "test"]:
    if split not in ds:
        continue
    for item in ds[split]:
        output.append({
            "dialogue_id": item.get("dialogue_id", ""),
            "turns": item.get("turns", {}),
            "services": item.get("services", []),
            "split": split
        })
print(json.dumps(output))
"""],
            capture_output=True, text=True,
            cwd=tempfile.gettempdir(),
            env={**os.environ, "PYTHONPATH": ""}
        )
        
        if result.returncode != 0:
            raise ImportError(f"Failed to load from HuggingFace: {result.stderr}")
        
        items = json.loads(result.stdout)
        
        sessions = []
        for item in items:
            session = self._convert_dialogue(item)
            if session:
                sessions.append(session)
        
        return sessions
    
    async def _load_from_sample(self, sample_path: str) -> List[Session]:
        """Load from sample_dialogues.json file."""
        with open(sample_path, "r") as f:
            data = json.load(f)
        
        sessions = []
        if isinstance(data, list):
            for item in data:
                session = self._convert_sample_dialogue(item)
                if session:
                    sessions.append(session)
        return sessions
    
    def _convert_sample_dialogue(self, item: Dict[str, Any]) -> Optional[Session]:
        """Convert a sample dialogue item to Session."""
        try:
            dialogue_id = item.get("dialogue_id", item.get("session_id", str(id(item))))
            turns_data = item.get("turns", [])
            
            turns = []
            for i, turn in enumerate(turns_data):
                turns.append(Turn(
                    turn_number=i + 1,
                    user_message=turn.get("user", turn.get("user_message", "")),
                    agent_response=turn.get("system", turn.get("agent_response", turn.get("agent", "")))
                ))
            
            if not turns:
                return None
            
            return Session(
                session_id=dialogue_id,
                turns=turns,
                metadata={
                    "source": "sample",
                    "domains": item.get("services", item.get("domains", []))
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert sample dialogue: {e}")
            return None
    
    async def _load_from_local(self) -> List[Session]:
        """Load dataset from local JSON files."""
        sessions = []
        
        # Look for JSON files in data directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, dict):
                    for dialogue_id, dialogue_data in data.items():
                        session = self._convert_dialogue_dict(
                            dialogue_id, dialogue_data
                        )
                        if session:
                            sessions.append(session)
                elif isinstance(data, list):
                    for item in data:
                        session = self._convert_dialogue(item)
                        if session:
                            sessions.append(session)
        
        return sessions
    
    def _convert_dialogue(self, item: Dict[str, Any]) -> Optional[Session]:
        """Convert a HuggingFace dataset item to Session."""
        try:
            dialogue_id = item.get("dialogue_id", str(id(item)))
            
            # Get turns from the item
            turns_data = item.get("turns", {})
            
            # Extract user and system utterances
            user_utterances = turns_data.get("utterance", [])
            speakers = turns_data.get("speaker", [])
            
            if not user_utterances:
                return None
            
            turns = []
            user_msg = None
            turn_number = 1
            
            for i, (utterance, speaker) in enumerate(zip(user_utterances, speakers)):
                if speaker == 0:  # User
                    user_msg = utterance
                else:  # System
                    if user_msg is not None:
                        turns.append(Turn(
                            turn_number=turn_number,
                            user_message=user_msg,
                            agent_response=utterance
                        ))
                        turn_number += 1
                        user_msg = None
            
            if not turns:
                return None
            
            # Extract domains from services
            services = item.get("services", [])
            
            return Session(
                session_id=dialogue_id,
                turns=turns,
                metadata={
                    "source": "multiwoz",
                    "domains": services,
                    "split": item.get("split", "unknown")
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert dialogue: {e}")
            return None
    
    def _convert_dialogue_dict(
        self, 
        dialogue_id: str, 
        data: Dict[str, Any]
    ) -> Optional[Session]:
        """Convert a dictionary-format dialogue to Session."""
        try:
            log = data.get("log", [])
            
            if not log:
                return None
            
            turns = []
            turn_number = 1
            
            # Log alternates between user and system
            for i in range(0, len(log) - 1, 2):
                user_turn = log[i]
                system_turn = log[i + 1] if i + 1 < len(log) else None
                
                if system_turn:
                    turns.append(Turn(
                        turn_number=turn_number,
                        user_message=user_turn.get("text", ""),
                        agent_response=system_turn.get("text", "")
                    ))
                    turn_number += 1
            
            if not turns:
                return None
            
            return Session(
                session_id=dialogue_id,
                turns=turns,
                metadata={
                    "source": "multiwoz",
                    "domains": data.get("goal", {}).get("domain", [])
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert dialogue {dialogue_id}: {e}")
            return None
    
    async def load_sample(
        self,
        sample_size: int,
        random_seed: Optional[int] = None,
        domains: Optional[List[str]] = None
    ) -> List[Session]:
        """
        Load a sample of dialogues, optionally filtered by domain.
        
        Args:
            sample_size: Number of dialogues to sample
            random_seed: Seed for reproducibility
            domains: Optional list of domains to filter by
            
        Returns:
            List of sampled Session objects
        """
        if not self._loaded:
            await self.load()
        
        sessions = self._sessions
        
        # Filter by domain if specified
        if domains:
            sessions = [
                s for s in sessions
                if any(d in s.metadata.get("domains", []) for d in domains)
            ]
        
        if sample_size >= len(sessions):
            return sessions
        
        if random_seed is not None:
            random.seed(random_seed)
        
        return random.sample(sessions, sample_size)
