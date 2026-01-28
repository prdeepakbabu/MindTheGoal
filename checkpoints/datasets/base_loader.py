"""Base class for dataset loaders."""

from abc import ABC, abstractmethod
from typing import List, Optional
import random

from core.models import Session


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders."""
    
    name: str = "base"
    description: str = "Base dataset loader"
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize loader with optional data directory."""
        self.data_dir = data_dir
        self._sessions: List[Session] = []
        self._loaded = False
    
    @abstractmethod
    async def load(self) -> List[Session]:
        """Load all sessions from the dataset."""
        pass
    
    async def load_sample(
        self,
        sample_size: int,
        random_seed: Optional[int] = None
    ) -> List[Session]:
        """
        Load a random sample of sessions.
        
        Args:
            sample_size: Number of sessions to sample
            random_seed: Optional seed for reproducibility
            
        Returns:
            List of sampled Session objects
        """
        # Load full dataset if not already loaded
        if not self._loaded:
            await self.load()
        
        if sample_size >= len(self._sessions):
            return self._sessions
        
        # Set seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        return random.sample(self._sessions, sample_size)
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded dataset."""
        if not self._sessions:
            return {"loaded": False, "sessions": 0, "turns": 0}
        
        total_turns = sum(s.total_turns for s in self._sessions)
        
        return {
            "loaded": True,
            "sessions": len(self._sessions),
            "turns": total_turns,
            "avg_turns_per_session": total_turns / len(self._sessions) if self._sessions else 0
        }
