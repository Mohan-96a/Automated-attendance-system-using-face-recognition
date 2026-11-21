"""
Face Enrollment and Biometric Onboarding Module for SmartAttend AI.
Validates quality, confirms liveness, extracts normalized 512-d embeddings,
encrypts templates, and prepares student biometric profiles.
"""

import os
import base64
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ai_engine.config.ai_config import AIConfig, get_ai_config
from ai_engine.detection.face_detector import FaceDetector, DetectedFace
from ai_engine.recognition.face_recognizer import FaceRecognizer, RegisteredPerson
from ai_engine.anti_spoofing.liveness_detector import LivenessDetector

logger = logging.getLogger("SmartAttend.FaceEnrollment")


@dataclass
class EnrollmentResult:
    success: bool
    student_id: str
    name: str
    message: str
    encrypted_embedding: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    quality_score: float = 0.0
    liveness_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)


class FaceEnrollmentManager:
    """
    Manages the complete enrollment workflow for students:
    detection -> quality filter -> liveness check -> alignment -> embedding generation -> encryption.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        liveness_detector: Optional[LivenessDetector] = None,
    ):
        self.config = config or get_ai_config()
        self.detector = detector or FaceDetector(self.config)
        self.recognizer = recognizer or FaceRecognizer(self.config)
        self.liveness_detector = liveness_detector or LivenessDetector(self.config)

    def _encode_thumbnail(self, face_crop: np.ndarray) -> str:
        """Converts face crop image into base64 JPEG thumbnail."""
        resized = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def enroll_from_image(
        self,
        student_id: str,
        name: str,
        image_or_path: Any,
        metadata: Optional[Dict[str, Any]] = None,
        enforce_liveness: bool = True,
    ) -> EnrollmentResult:
        """
        Enrolls a student from an image file path or BGR numpy array.
        """
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                return EnrollmentResult(
                    success=False,
                    student_id=student_id,
                    name=name,
                    message=f"Image file not found: {image_or_path}",
                    validation_errors=["File does not exist"],
                )
            frame = cv2.imread(image_or_path)
            if frame is None:
                return EnrollmentResult(
                    success=False,
                    student_id=student_id,
                    name=name,
                    message="Failed to decode image file",
                    validation_errors=["Corrupt or invalid image format"],
                )
        elif isinstance(image_or_path, np.ndarray):
            frame = image_or_path
        else:
            return EnrollmentResult(
                success=False,
                student_id=student_id,
                name=name,
                message="Invalid image input type",
                validation_errors=["Input must be filepath or numpy ndarray"],
            )

        # 1. Face Detection
        detected_faces = self.detector.detect_faces(frame)
        if len(detected_faces) == 0:
            return EnrollmentResult(
                success=False,
                student_id=student_id,
                name=name,
                message="No face detected in the provided image. Ensure the face is clearly visible.",
                validation_errors=["No face detected"],
            )

        if len(detected_faces) > 1:
            return EnrollmentResult(
                success=False,
                student_id=student_id,
                name=name,
                message="Multiple faces detected. Please provide an image with only the student's face.",
                validation_errors=["Multiple faces detected"],
            )

        face = detected_faces[0]

        # 2. Quality Assessment
        validation_errors = []
        if not face.quality.is_acceptable:
            validation_errors.extend(face.quality.rejection_reasons)

        # 3. Liveness / Anti-Spoofing Check
        liveness_res = self.liveness_detector.verify_liveness(
            face.raw_face_crop,
            landmarks=face.landmarks,
            yaw_angle=face.quality.yaw_angle,
        )

        if enforce_liveness and not liveness_res.is_live:
            validation_errors.append(
                f"Liveness verification failed ({liveness_res.spoof_type or 'Possible spoof'})"
            )

        if validation_errors:
            return EnrollmentResult(
                success=False,
                student_id=student_id,
                name=name,
                message="Face image does not satisfy enrollment standards.",
                quality_score=face.quality.blur_score,
                liveness_score=liveness_res.liveness_score,
                validation_errors=validation_errors,
            )

        # 4. Generate Embedding and Register
        registered = self.recognizer.register_face(
            student_id=student_id,
            name=name,
            aligned_face=face.aligned_face,
            metadata=metadata or {},
        )

        thumb_b64 = self._encode_thumbnail(face.raw_face_crop)

        logger.info("Successfully enrolled student %s (%s)", name, student_id)
        return EnrollmentResult(
            success=True,
            student_id=student_id,
            name=name,
            message="Student face enrolled successfully with encrypted biometric template.",
            encrypted_embedding=registered.encrypted_embedding,
            thumbnail_base64=thumb_b64,
            quality_score=face.quality.blur_score,
            liveness_score=liveness_res.liveness_score,
            validation_errors=[],
        )
