"""
AI Engine for SmartAttend AI
Modular Computer Vision, Face Recognition, Anti-Spoofing and Face Tracking Engine
"""

from ai_engine.config.ai_config import AIConfig, get_ai_config
from ai_engine.detection.face_detector import FaceDetector, DetectedFace
from ai_engine.recognition.face_recognizer import FaceRecognizer, RecognitionResult
from ai_engine.anti_spoofing.liveness_detector import LivenessDetector, LivenessResult
from ai_engine.tracking.face_tracker import FaceTracker, TrackedFace
from ai_engine.enrollment.face_enrollment import FaceEnrollmentManager
from ai_engine.engine import SmartAttendAIEngine

__all__ = [
    "AIConfig",
    "get_ai_config",
    "FaceDetector",
    "DetectedFace",
    "FaceRecognizer",
    "RecognitionResult",
    "LivenessDetector",
    "LivenessResult",
    "FaceTracker",
    "TrackedFace",
    "FaceEnrollmentManager",
    "SmartAttendAIEngine",
]
