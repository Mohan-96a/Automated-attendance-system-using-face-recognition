"""
SmartAttend AI - Main FastAPI Application.
Coordinates REST API routes, WebSockets, Lifespan initialization, and AI integration.
"""

import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.database.mongodb import db_manager
from backend.app.database.seed_data import seed_initial_data
from backend.app.services.attendance_service import AttendanceService

# Routers
from backend.app.routes.auth_routes import router as auth_router
from backend.app.routes.student_routes import router as student_router
from backend.app.routes.faculty_routes import router as faculty_router
from backend.app.routes.academic_routes import router as academic_router
from backend.app.routes.attendance_routes import router as attendance_router
from backend.app.routes.setting_routes import router as setting_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SmartAttend.App")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database connection and seeds demo data on application startup."""
    logger.info("Initializing SmartAttend AI Backend services...")
    await db_manager.connect()
    await seed_initial_data()
    yield
    logger.info("SmartAttend AI Backend shut down successfully.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent AI-Powered Smart Attendance Management System with ArcFace Facial Recognition, Liveness Detection, and Role-Based Dashboards.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular API V1 routers
api_prefix = settings.API_V1_PREFIX
app.include_router(auth_router, prefix=api_prefix)
app.include_router(student_router, prefix=api_prefix)
app.include_router(faculty_router, prefix=api_prefix)
app.include_router(academic_router, prefix=api_prefix)
app.include_router(attendance_router, prefix=api_prefix)
app.include_router(setting_router, prefix=api_prefix)


@app.get("/", tags=["Health"])
async def root():
    return {
        "product": "SmartAttend AI",
        "version": settings.VERSION,
        "status": "ONLINE",
        "docs_url": "/docs",
        "database_connected": db_manager.is_connected,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "database": "mongodb" if db_manager.is_connected else "local_json_resilient_engine",
        "ai_engine": "online",
    }


# --- WebSocket for Real-Time Live Attendance Camera Feed ---
class AttendanceConnectionManager:
    """Manages active WebSocket camera streams per attendance session."""

    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(session_id, []).append(websocket)
        logger.info("WebSocket client connected to session: %s", session_id)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info("WebSocket client disconnected from session: %s", session_id)

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning("Error broadcasting WebSocket frame: %s", e)


ws_manager = AttendanceConnectionManager()


@app.websocket("/ws/attendance/{session_id}")
async def attendance_camera_stream(websocket: WebSocket, session_id: str):
    """
    Real-time bidirectional WebSocket endpoint:
    Receives webcam frames from frontend client -> processes through AI Engine -> returns overlay and recognition events.
    """
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                payload = json.loads(data_text)
                frame_b64 = payload.get("frame")
                user_lat = payload.get("lat")
                user_lon = payload.get("lon")

                if frame_b64:
                    result = await AttendanceService.process_live_frame(
                        session_id=session_id,
                        frame_base64=frame_b64,
                        user_lat=user_lat,
                        user_lon=user_lon,
                    )
                    await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({"error": str(e), "success": False})
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id, websocket)
    except Exception as e:
        logger.error("WebSocket exception: %s", e)
        ws_manager.disconnect(session_id, websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
