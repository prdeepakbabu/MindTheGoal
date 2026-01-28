"""Evaluation API routes."""

import logging
import json
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.models import Session, Turn, EvaluationResult, TurnQuality
from core.gsr_calculator import GSRCalculator
from core.goal_segmentation import GoalSegmenter
from agents.judge_agent import JudgeAgent
from datasets.registry import get_loader
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class EvaluateDatasetRequest(BaseModel):
    """Request to evaluate a dataset."""
    dataset: str = Field(description="Dataset name: multiwoz, sgd, or custom")
    sample_size: Optional[int] = Field(default=None, description="Number of dialogues to sample (None for full dataset)")
    random_seed: Optional[int] = Field(default=42, description="Random seed for sampling")
    data_dir: Optional[str] = Field(default=None, description="Custom data directory")


class EvaluateSessionRequest(BaseModel):
    """Request to evaluate a single session."""
    turns: List[dict] = Field(description="List of turn objects with user_message and agent_response")
    session_id: Optional[str] = Field(default=None, description="Optional session ID")


class TurnInput(BaseModel):
    """Single turn input for evaluation."""
    user_message: str
    agent_response: str


class GSRResponse(BaseModel):
    """GSR evaluation response."""
    evaluation_id: str
    dataset_name: str
    overall_gsr: float
    single_turn_gsr: float
    multi_turn_gsr: float
    total_goals: int
    total_sessions: int
    total_turns: int
    successful_goals: int
    failed_goals: int
    rcof_distribution: dict


class SessionEvaluationResponse(BaseModel):
    """Single session evaluation response."""
    session_id: str
    gsr: float
    total_goals: int
    total_turns: int
    goals: List[dict]


# In-memory storage for evaluation jobs
_evaluation_jobs = {}


@router.post("/dataset", response_model=GSRResponse)
async def evaluate_dataset(request: EvaluateDatasetRequest):
    """
    Evaluate a dataset and calculate GSR metrics.
    
    Supports MultiWOZ, SGD, and custom JSON datasets.
    Use sample_size to limit the number of dialogues evaluated.
    """
    settings = get_settings()
    
    try:
        # Get dataset loader
        loader = get_loader(
            name=request.dataset,
            data_dir=request.data_dir or settings.get_datasets_path(request.dataset)
        )
        
        # Load sessions
        if request.sample_size:
            sessions = await loader.load_sample(
                sample_size=request.sample_size,
                random_seed=request.random_seed
            )
        else:
            sessions = await loader.load()
        
        logger.info(f"Loaded {len(sessions)} sessions from {request.dataset}")
        
        # Initialize components
        judge = JudgeAgent()
        segmenter = GoalSegmenter()
        calculator = GSRCalculator()
        
        # Evaluate each session
        evaluated_sessions = []
        for session in sessions:
            # Evaluate turns with judge
            evaluations = await judge.evaluate_session(session)
            session = await judge.apply_evaluations(session, evaluations)
            
            # Segment into goals
            session = segmenter.segment_session(session)
            
            evaluated_sessions.append(session)
        
        # Calculate GSR metrics
        result = calculator.calculate_dataset_gsr(evaluated_sessions)
        result.dataset_name = request.dataset
        
        # Generate report
        report = calculator.generate_report(result)
        
        return GSRResponse(
            evaluation_id=report.evaluation_id,
            dataset_name=report.dataset_name,
            overall_gsr=report.overall_gsr,
            single_turn_gsr=report.single_turn_gsr,
            multi_turn_gsr=report.multi_turn_gsr,
            total_goals=report.total_goals,
            total_sessions=report.total_sessions,
            total_turns=report.total_turns,
            successful_goals=report.successful_goals,
            failed_goals=report.failed_goals,
            rcof_distribution=report.rcof_distribution
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/session", response_model=SessionEvaluationResponse)
async def evaluate_session(request: EvaluateSessionRequest):
    """
    Evaluate a single conversation session.
    
    Provides detailed goal-level breakdown with RCOF for failures.
    """
    try:
        # Convert input to turns
        turns = []
        for i, turn_data in enumerate(request.turns):
            turns.append(Turn(
                turn_number=i + 1,
                user_message=turn_data.get("user_message", ""),
                agent_response=turn_data.get("agent_response", "")
            ))
        
        session = Session(
            session_id=request.session_id or "manual-session",
            turns=turns
        )
        
        # Evaluate
        judge = JudgeAgent()
        segmenter = GoalSegmenter()
        
        evaluations = await judge.evaluate_session(session)
        session = await judge.apply_evaluations(session, evaluations)
        session = segmenter.segment_session(session)
        
        return SessionEvaluationResponse(
            session_id=session.session_id,
            gsr=session.gsr,
            total_goals=session.total_goals,
            total_turns=session.total_turns,
            goals=[g.to_dict() for g in session.goals]
        )
        
    except Exception as e:
        logger.error(f"Session evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/turn")
async def evaluate_single_turn(
    user_message: str,
    agent_response: str,
    conversation_history: Optional[List[dict]] = None
):
    """
    Evaluate a single turn in real-time.
    
    Useful for live chat evaluation.
    """
    try:
        judge = JudgeAgent()
        
        result = await judge.evaluate_single_response(
            user_message=user_message,
            agent_response=agent_response,
            conversation_history=conversation_history
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Turn evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rcof-codes")
async def get_rcof_codes():
    """Get all RCOF codes with descriptions."""
    from core.models import RCOF
    return RCOF.get_all_descriptions()


@router.get("/results")
async def list_results():
    """List available evaluation result files."""
    project_root = Path(__file__).parent.parent.parent
    results = []
    for f in project_root.glob("*_results.json"):
        results.append({
            "filename": f.name,
            "path": f"/api/evaluation/results/{f.name}",
            "size": f.stat().st_size
        })
    return {"results": results}


@router.get("/results/{filename}")
async def get_results(filename: str):
    """Get a specific evaluation results file."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Results file not found: {filename}")
    
    if not filename.endswith("_results.json"):
        raise HTTPException(status_code=400, detail="Invalid results file name")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")


@router.get("/logs")
async def list_log_dirs():
    """List available evaluation log directories."""
    project_root = Path(__file__).parent.parent.parent
    logs_path = project_root / "logs" / "evaluations"
    
    if not logs_path.exists():
        return {"log_dirs": []}
    
    dirs = []
    for d in logs_path.iterdir():
        if d.is_dir():
            session_count = len(list(d.glob("session_*.json")))
            dirs.append({
                "name": d.name,
                "session_count": session_count
            })
    
    return {"log_dirs": sorted(dirs, key=lambda x: x["name"], reverse=True)}


@router.get("/logs/{log_dir}/session/{session_index}")
async def get_session_log(log_dir: str, session_index: int):
    """Get detailed session log by index."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / "logs" / "evaluations" / log_dir / f"session_{session_index}.json"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Session log not found: {log_dir}/session_{session_index}.json")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading session log: {str(e)}")


@router.get("/logs/{log_dir}/session-by-id/{session_id}")
async def get_session_by_id(log_dir: str, session_id: str):
    """Get detailed session log by session ID."""
    project_root = Path(__file__).parent.parent.parent
    logs_path = project_root / "logs" / "evaluations" / log_dir
    
    if not logs_path.exists():
        raise HTTPException(status_code=404, detail=f"Log directory not found: {log_dir}")
    
    # Search through all session files
    for f in logs_path.glob("session_*.json"):
        try:
            with open(f, "r") as file:
                data = json.load(file)
                if data.get("session_id") == session_id:
                    return JSONResponse(content=data)
        except:
            continue
    
    raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


class StartEvaluationRequest(BaseModel):
    """Request to start an evaluation from the frontend."""
    dataset: str = Field(description="Dataset name or 'custom'")
    sample_size: int = Field(default=10, description="Number of dialogues to evaluate")
    name: str = Field(default="", description="Optional name for the evaluation")


_running_evaluations = {}


@router.post("/start")
async def start_evaluation(request: StartEvaluationRequest, background_tasks: BackgroundTasks):
    """Start a new evaluation (runs in background)."""
    import uuid
    import subprocess
    import sys
    
    eval_id = str(uuid.uuid4())[:8]
    
    # Run evaluation using CLI in background
    project_root = Path(__file__).parent.parent.parent
    
    _running_evaluations[eval_id] = {
        "status": "running",
        "dataset": request.dataset,
        "sample_size": request.sample_size,
        "started_at": datetime.now().isoformat()
    }
    
    async def run_eval():
        try:
            import asyncio
            from datasets.registry import get_loader
            from core.gsr_calculator import GSRCalculator
            from core.goal_segmentation import GoalSegmenter
            from agents.judge_agent import JudgeAgent
            
            settings = get_settings()
            
            # Get dataset loader
            loader = get_loader(
                name=request.dataset,
                data_dir=settings.get_datasets_path(request.dataset)
            )
            
            # Load sessions
            sessions = await loader.load_sample(
                sample_size=request.sample_size,
                random_seed=42
            )
            
            logger.info(f"Loaded {len(sessions)} sessions for evaluation {eval_id}")
            
            # Initialize components
            judge = JudgeAgent()
            segmenter = GoalSegmenter()
            calculator = GSRCalculator()
            
            # Evaluate each session
            evaluated_sessions = []
            for i, session in enumerate(sessions):
                _running_evaluations[eval_id]["progress"] = f"{i+1}/{len(sessions)}"
                
                # Evaluate turns with judge
                evaluations = await judge.evaluate_session(session)
                session = await judge.apply_evaluations(session, evaluations)
                
                # Segment into goals
                session = segmenter.segment_session(session)
                
                evaluated_sessions.append(session)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(10)
            
            # Calculate GSR metrics
            result = calculator.calculate_dataset_gsr(evaluated_sessions)
            result.dataset_name = request.dataset
            
            # Save results
            name = request.name or f"{request.dataset}_{request.sample_size}"
            output_file = project_root / f"{name}_results.json"
            
            report = calculator.generate_report(result)
            report_dict = {
                "evaluation_id": report.evaluation_id,
                "dataset_name": report.dataset_name,
                "total_sessions": report.total_sessions,
                "total_goals": report.total_goals,
                "total_turns": report.total_turns,
                "overall_gsr": report.overall_gsr,
                "single_turn_gsr": report.single_turn_gsr,
                "multi_turn_gsr": report.multi_turn_gsr,
                "rcof_distribution": report.rcof_distribution,
                "domain_gsr": getattr(report, 'domain_gsr', {}),
                "evaluated_at": report.evaluated_at if isinstance(getattr(report, 'evaluated_at', None), str) else (report.evaluated_at.isoformat() if hasattr(report, 'evaluated_at') and report.evaluated_at else datetime.now().isoformat()),
                "session_details": [
                    {
                        "session_id": s.session_id,
                        "num_turns": len(s.turns),
                        "num_goals": len(s.goals),
                        "goals": [
                            {
                                "goal_number": g.goal_number,
                                "num_turns": len(g.turns),
                                "quality": g.quality.value if hasattr(g.quality, 'value') else str(g.quality),
                                "rcof": (g.rcof.value if hasattr(g.rcof, 'value') else str(g.rcof)) if g.rcof else None
                            }
                            for g in s.goals
                        ]
                    }
                    for s in evaluated_sessions
                ]
            }
            
            with open(output_file, "w") as f:
                json.dump(report_dict, f, indent=2)
            
            _running_evaluations[eval_id] = {
                "status": "completed",
                "dataset": request.dataset,
                "sample_size": request.sample_size,
                "overall_gsr": report.overall_gsr,
                "output_file": str(output_file.name),
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evaluation {eval_id} failed: {e}")
            _running_evaluations[eval_id] = {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
    
    background_tasks.add_task(run_eval)
    
    return {
        "evaluation_id": eval_id,
        "status": "started",
        "message": f"Evaluation started for {request.sample_size} {request.dataset} dialogues"
    }


@router.get("/status/{eval_id}")
async def get_evaluation_status(eval_id: str):
    """Get status of a running evaluation."""
    if eval_id not in _running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return _running_evaluations[eval_id]
