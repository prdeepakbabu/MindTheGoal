"""Schema-Guided Dialogue (SGD) dataset loader."""

import os
import json
import random
from typing import List, Optional, Dict, Any
import logging

from datasets.base_loader import BaseDatasetLoader
from core.models import Session, Turn

logger = logging.getLogger(__name__)


class SGDLoader(BaseDatasetLoader):
    """
    Loader for Schema-Guided Dialogue (SGD) dataset.
    
    SGD is Google's task-oriented dialogue dataset with ~20,000 dialogues
    across 20+ domains/services.
    """
    
    name = "sgd"
    description = "Schema-Guided Dialogue - Multi-domain task-oriented dialogues"
    
    # Hugging Face dataset name
    HF_DATASET = "schema_guided_dstc8"
    
    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self._raw_data: Dict[str, Any] = {}
    
    async def load(self) -> List[Session]:
        """Load SGD dataset."""
        logger.info("Loading SGD dataset...")
        
        try:
            sessions = await self._load_from_huggingface()
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            if self.data_dir and os.path.exists(self.data_dir):
                sessions = await self._load_from_local()
            else:
                raise RuntimeError(
                    "Could not load SGD. Install 'datasets' package or "
                    "provide local data directory."
                )
        
        self._sessions = sessions
        self._loaded = True
        logger.info(f"Loaded {len(sessions)} sessions from SGD")
        return sessions
    
    async def _load_from_huggingface(self) -> List[Session]:
        """Load dataset from Hugging Face."""
        from datasets import load_dataset
        
        dataset = load_dataset(self.HF_DATASET, trust_remote_code=True)
        
        sessions = []
        
        for split in ["train", "validation", "test"]:
            if split not in dataset:
                continue
            
            for item in dataset[split]:
                session = self._convert_dialogue(item, split)
                if session:
                    sessions.append(session)
        
        return sessions
    
    async def _load_from_local(self) -> List[Session]:
        """Load dataset from local JSON files."""
        sessions = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        session = self._convert_dialogue(item)
                        if session:
                            sessions.append(session)
        
        return sessions
    
    def _convert_dialogue(
        self, 
        item: Dict[str, Any],
        split: str = "unknown"
    ) -> Optional[Session]:
        """Convert SGD dialogue to Session format."""
        try:
            dialogue_id = item.get("dialogue_id", str(id(item)))
            
            turns_data = item.get("turns", [])
            
            if not turns_data:
                return None
            
            turns = []
            turn_number = 1
            user_msg = None
            
            for turn_item in turns_data:
                speaker = turn_item.get("speaker", "")
                utterance = turn_item.get("utterance", "")
                
                if speaker.upper() == "USER":
                    user_msg = utterance
                elif speaker.upper() == "SYSTEM" and user_msg is not None:
                    turns.append(Turn(
                        turn_number=turn_number,
                        user_message=user_msg,
                        agent_response=utterance
                    ))
                    turn_number += 1
                    user_msg = None
            
            if not turns:
                return None
            
            services = item.get("services", [])
            
            return Session(
                session_id=dialogue_id,
                turns=turns,
                metadata={
                    "source": "sgd",
                    "domains": services,
                    "split": split
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert SGD dialogue: {e}")
            return None
    
    async def load_sample(
        self,
        sample_size: int,
        random_seed: Optional[int] = None,
        services: Optional[List[str]] = None
    ) -> List[Session]:
        """
        Load a sample of dialogues, optionally filtered by service.
        
        Args:
            sample_size: Number of dialogues to sample
            random_seed: Seed for reproducibility  
            services: Optional list of services to filter by
            
        Returns:
            List of sampled Session objects
        """
        if not self._loaded:
            await self.load()
        
        sessions = self._sessions
        
        if services:
            sessions = [
                s for s in sessions
                if any(svc in s.metadata.get("domains", []) for svc in services)
            ]
        
        if sample_size >= len(sessions):
            return sessions
        
        if random_seed is not None:
            random.seed(random_seed)
        
        return random.sample(sessions, sample_size)
