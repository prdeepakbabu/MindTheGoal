"""Chat API routes with real-time evaluation."""

import logging
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from agents.chat_agent import ChatAgent, TaskOrientedChatAgent
from agents.judge_agent import JudgeAgent

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request with message."""
    message: str = Field(description="User's message")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    domain: Optional[str] = Field(default=None, description="Task domain (restaurant, hotel, train, taxi, attraction)")
    evaluate: bool = Field(default=True, description="Whether to evaluate the response")


class ChatResponse(BaseModel):
    """Chat response with evaluation."""
    response: str
    session_id: str
    turn_number: int
    evaluation: Optional[dict] = None


# In-memory session storage
_chat_sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(session_id: Optional[str], domain: Optional[str] = None) -> tuple:
    """Get existing session or create new one."""
    import uuid
    
    if session_id and session_id in _chat_sessions:
        return session_id, _chat_sessions[session_id]
    
    new_id = session_id or str(uuid.uuid4())
    
    # Create appropriate chat agent
    if domain:
        chat_agent = TaskOrientedChatAgent(domain=domain)
    else:
        chat_agent = ChatAgent()
    
    _chat_sessions[new_id] = {
        "chat_agent": chat_agent,
        "history": [],
        "turn_number": 0,
        "domain": domain
    }
    
    return new_id, _chat_sessions[new_id]


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message and get a response with evaluation.
    
    Automatically evaluates the response if evaluate=True.
    """
    try:
        session_id, session = get_or_create_session(request.session_id, request.domain)
        
        chat_agent = session["chat_agent"]
        history = session["history"]
        session["turn_number"] += 1
        turn_number = session["turn_number"]
        
        # Generate response
        response = await chat_agent.respond(
            user_message=request.message,
            conversation_history=history
        )
        
        # Evaluate if requested
        evaluation = None
        if request.evaluate:
            judge = JudgeAgent()
            evaluation = await judge.evaluate_single_response(
                user_message=request.message,
                agent_response=response,
                conversation_history=history
            )
        
        # Update history
        history.append({
            "user": request.message,
            "agent": response
        })
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            turn_number=turn_number,
            evaluation=evaluation
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with evaluation.
    
    Supports streaming responses and live evaluation updates.
    """
    await websocket.accept()
    
    session_id, session = get_or_create_session(session_id)
    chat_agent = session["chat_agent"]
    judge = JudgeAgent()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            evaluate = message_data.get("evaluate", True)
            
            session["turn_number"] += 1
            turn_number = session["turn_number"]
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "ack",
                "turn_number": turn_number
            })
            
            # Stream response
            response_chunks = []
            async for chunk in chat_agent.respond_streaming(
                user_message=user_message,
                conversation_history=session["history"]
            ):
                response_chunks.append(chunk)
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk,
                    "turn_number": turn_number
                })
            
            full_response = "".join(response_chunks)
            
            # Send complete response
            await websocket.send_json({
                "type": "response",
                "content": full_response,
                "turn_number": turn_number
            })
            
            # Evaluate if requested
            if evaluate:
                evaluation = await judge.evaluate_single_response(
                    user_message=user_message,
                    agent_response=full_response,
                    conversation_history=session["history"]
                )
                
                await websocket.send_json({
                    "type": "evaluation",
                    "data": evaluation,
                    "turn_number": turn_number
                })
            
            # Update history
            session["history"].append({
                "user": user_message,
                "agent": full_response
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1001)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and history."""
    if session_id not in _chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _chat_sessions[session_id]
    return {
        "session_id": session_id,
        "turn_count": session["turn_number"],
        "domain": session.get("domain"),
        "history": session["history"]
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in _chat_sessions:
        del _chat_sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/domains")
async def get_available_domains():
    """Get available task domains."""
    return {
        "domains": list(TaskOrientedChatAgent.DOMAIN_PROMPTS.keys()),
        "default": "general"
    }
