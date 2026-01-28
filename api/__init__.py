"""FastAPI backend for MindTheGoal evaluation framework."""

from api.main import app, create_app
from api.routes import evaluation, chat, datasets

__all__ = [
    "app",
    "create_app",
    "evaluation",
    "chat", 
    "datasets",
]
