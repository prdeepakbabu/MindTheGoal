"""Dataset API routes."""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from datasets.registry import DatasetRegistry, get_loader
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class DatasetInfo(BaseModel):
    """Dataset information response."""
    name: str
    description: str
    loaded: bool = False
    session_count: int = 0
    turn_count: int = 0


class DatasetStatsResponse(BaseModel):
    """Dataset statistics response."""
    name: str
    total_sessions: int
    total_turns: int
    avg_turns_per_session: float
    sample_sessions: Optional[List[dict]] = None


@router.get("/available")
async def list_available_datasets():
    """List all available dataset loaders."""
    return {
        "datasets": DatasetRegistry.list_available()
    }


@router.get("/{dataset_name}/stats")
async def get_dataset_stats(
    dataset_name: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42
):
    """
    Get statistics for a dataset.
    
    Optionally sample a subset of the dataset.
    """
    try:
        settings = get_settings()
        loader = get_loader(
            name=dataset_name,
            data_dir=settings.get_datasets_path(dataset_name)
        )
        
        if sample_size:
            sessions = await loader.load_sample(
                sample_size=sample_size,
                random_seed=random_seed
            )
        else:
            sessions = await loader.load()
        
        total_turns = sum(s.total_turns for s in sessions)
        
        return DatasetStatsResponse(
            name=dataset_name,
            total_sessions=len(sessions),
            total_turns=total_turns,
            avg_turns_per_session=total_turns / len(sessions) if sessions else 0
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/sample")
async def get_dataset_sample(
    dataset_name: str,
    sample_size: int = 5,
    random_seed: int = 42
):
    """
    Get a sample of dialogues from a dataset.
    
    Useful for previewing dataset contents.
    """
    try:
        settings = get_settings()
        loader = get_loader(
            name=dataset_name,
            data_dir=settings.get_datasets_path(dataset_name)
        )
        
        sessions = await loader.load_sample(
            sample_size=sample_size,
            random_seed=random_seed
        )
        
        return {
            "dataset": dataset_name,
            "sample_size": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "turn_count": s.total_turns,
                    "metadata": s.metadata,
                    "turns": [
                        {
                            "turn_number": t.turn_number,
                            "user_message": t.user_message[:200] + "..." if len(t.user_message) > 200 else t.user_message,
                            "agent_response": t.agent_response[:200] + "..." if len(t.agent_response) > 200 else t.agent_response
                        }
                        for t in s.turns
                    ]
                }
                for s in sessions
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get dataset sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/session/{session_id}")
async def get_session_by_id(
    dataset_name: str,
    session_id: str
):
    """
    Get a specific session from a dataset by ID.
    """
    try:
        settings = get_settings()
        loader = get_loader(
            name=dataset_name,
            data_dir=settings.get_datasets_path(dataset_name)
        )
        
        sessions = await loader.load()
        
        for session in sessions:
            if session.session_id == session_id:
                return {
                    "session_id": session.session_id,
                    "metadata": session.metadata,
                    "turns": [t.to_dict() for t in session.turns]
                }
        
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
