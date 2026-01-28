"""FastAPI application setup."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    # Startup
    logger.info("Starting MindTheGoal API server...")
    yield
    # Shutdown
    logger.info("Shutting down MindTheGoal API server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="MindTheGoal API",
        description="Goal Success Rate (GSR) evaluation framework for multi-turn conversations",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from api.routes import evaluation, chat, datasets as datasets_routes
    
    app.include_router(
        evaluation.router,
        prefix="/api/evaluation",
        tags=["Evaluation"]
    )
    app.include_router(
        chat.router,
        prefix="/api/chat",
        tags=["Chat"]
    )
    app.include_router(
        datasets_routes.router,
        prefix="/api/datasets",
        tags=["Datasets"]
    )
    
    # Serve static frontend files
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    @app.get("/")
    async def root():
        """Serve dashboard as homepage."""
        dashboard_path = Path(__file__).parent.parent / "frontend" / "dashboard.html"
        if dashboard_path.exists():
            return FileResponse(dashboard_path)
        return {
            "name": "MindTheGoal API",
            "version": "1.0.0",
            "description": "Goal Success Rate evaluation for multi-turn conversations",
            "docs": "/docs"
        }
    
    @app.get("/dashboard")
    async def dashboard():
        """Serve dashboard page."""
        return FileResponse(Path(__file__).parent.parent / "frontend" / "dashboard.html")
    
    @app.get("/evaluate")
    async def evaluate_page():
        """Serve evaluation runner page."""
        return FileResponse(Path(__file__).parent.parent / "frontend" / "evaluate.html")
    
    @app.get("/viewer")
    async def viewer_page():
        """Serve dialog viewer page."""
        return FileResponse(Path(__file__).parent.parent / "frontend" / "viewer.html")
    
    @app.get("/settings")
    async def settings_page():
        """Serve settings page."""
        return FileResponse(Path(__file__).parent.parent / "frontend" / "settings.html")
    
    @app.get("/chat")
    async def chat_page():
        """Serve chat demo page."""
        return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
