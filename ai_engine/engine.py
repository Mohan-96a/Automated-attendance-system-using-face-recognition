"""
Unified SmartAttend AI Engine.
Integrates Face Detection, Quality Assessment, Face Tracking,
Anti-Spoofing Liveness Verification, and Biometric Recognition into a single high-performance pipeline.
"""

import time
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ai_engine.config.ai_config import AIConfig, get_ai_config
from ai_engine.detection.face_detector import FaceDetector, DetectedFace
from ai_engine.tracking.face_tracker import FaceTracker, TrackedFace
from ai_engine.recognition.face_recognizer import FaceRecognizer, RecognitionResult
from ai_engine.anti_spoofing.liveness_detector import LivenessDetector, LivenessResult
from ai_engine.enrollment.face_enrollment import FaceEnrollmentManager

logger = logging.getLogger("SmartAttend.AIEngine")


@dataclass
class ProcessedFaceEvent:
    track_id: int
    bbox: Tuple[int, int, int, int]
    name: str
    student_id: Optional[str]
    confidence_percentage: float
    is_known: bool
    is_live: bool
    liveness_score: float
    distance_cm: int
    distance_status: str
    distance_color: Tuple[int, int, int]
    is_ready_for_attendance: bool
    rejection_reason: Optional[str] = None


@dataclass
class FrameProcessingResult:
    frame_timestamp: float
    total_faces_detected: int
    multiple_faces_detected: bool
    processed_faces: List[ProcessedFaceEvent] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    fps: float = 0.0


class SmartAttendAIEngine:
    """
    Central AI Engine coordinator.
    Processes live camera frames and produces structured recognition events and annotated overlays.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or get_ai_config()
        self.detector = FaceDetector(self.config)
        self.tracker = FaceTracker(self.config)
        self.recognizer = FaceRecognizer(self.config)
        self.liveness = LivenessDetector(self.config)
        self.enrollment = FaceEnrollmentManager(
            self.config, self.detector, self.recognizer, self.liveness
        )
        self.prev_frame_time = time.time()

    def process_frame(
        self, frame: np.ndarray, draw_overlay: bool = True
    ) -> FrameProcessingResult:
        """
        Executes end-to-end inference on a video frame:
        1. Multi-face detection & landmarks
        2. Centroid/IoU tracking
        3. Quality & anti-spoofing verification
        4. ArcFace recognition (with tracking cache & cooldown)
        5. Visual overlay drawing (optimal distance indicator, bounding boxes, labels)
        """
        start_time = time.time()
        if frame is None or frame.size == 0:
            return FrameProcessingResult(
                frame_timestamp=start_time,
                total_faces_detected=0,
                multiple_faces_detected=False,
            )

        annotated = frame.copy() if draw_overlay else None
        h, w = frame.shape[:2]

        # 1. Detect faces
        detected_faces = self.detector.detect_faces(frame)
        total_faces = len(detected_faces)
        is_multiple = total_faces > 1

        # 2. Update tracker
        tracked_faces = self.tracker.update(detected_faces)

        # Map bbox to detected face object for lookup
        det_map: Dict[Tuple[int, int, int, int], DetectedFace] = {
            det.bbox: det for det in detected_faces
        }

        face_events: List[ProcessedFaceEvent] = []

        for track in tracked_faces:
            det = det_map.get(track.bbox)
            raw_crop = det.raw_face_crop if det is not None else None
            aligned_face = det.aligned_face if det is not None else None
            quality = det.quality if det is not None else None

            # Check if this track needs recognition
            if track.needs_recognition and aligned_face is not None:
                # Anti-spoofing check
                if self.config.enable_anti_spoofing:
                    live_res = self.liveness.verify_liveness(
                        raw_crop,
                        landmarks=det.landmarks if det else None,
                        yaw_angle=quality.yaw_angle if quality else 0.0,
                    )
                    track.is_live = live_res.is_live
                    track.liveness_score = live_res.liveness_score

                # Face Recognition
                if track.is_live:
                    rec_res = self.recognizer.recognize(aligned_face, enforce_cooldown=True)
                    if rec_res.matched:
                        track.recognized_name = rec_res.name or "Student"
                        track.student_id = rec_res.student_id
                        track.confidence_percentage = rec_res.confidence_percentage
                        track.is_known = True
                    else:
                        track.recognized_name = self.config.unknown_label
                        track.student_id = None
                        track.confidence_percentage = rec_res.confidence_percentage
                        track.is_known = False
                else:
                    track.recognized_name = "Spoof Detected"
                    track.is_known = False

            # Determine if ready to record attendance:
            # Must be known, live, acceptable quality, and in "Good Distance"
            dist_ok = track.distance_status == "Good Distance"
            quality_ok = quality.is_acceptable if quality is not None else True

            is_ready = bool(
                track.is_known and
                track.is_live and
                dist_ok and
                quality_ok
            )

            rejection_reason = None
            if not track.is_known:
                rejection_reason = "Unrecognized face"
            elif not track.is_live:
                rejection_reason = "Liveness check failed"
            elif not dist_ok:
                rejection_reason = f"Distance warning: {track.distance_status}"
            elif not quality_ok and quality:
                rejection_reason = ", ".join(quality.rejection_reasons)

            dist_color = (0, 255, 0) if dist_ok else (0, 0, 255)
            if track.distance_status == "Move Closer":
                dist_color = (0, 165, 255)

            event = ProcessedFaceEvent(
                track_id=track.track_id,
                bbox=track.bbox,
                name=track.recognized_name,
                student_id=track.student_id,
                confidence_percentage=track.confidence_percentage,
                is_known=track.is_known,
                is_live=track.is_live,
                liveness_score=track.liveness_score,
                distance_cm=track.distance_cm,
                distance_status=track.distance_status,
                distance_color=dist_color,
                is_ready_for_attendance=is_ready,
                rejection_reason=rejection_reason,
            )
            face_events.append(event)

            # Draw visual overlay on frame
            if draw_overlay and annotated is not None:
                x, y, fw, fh = track.bbox

                # Face Bounding Box with distance-status color
                box_color = (0, 255, 0) if (track.is_known and track.is_live) else (0, 0, 255)
                if not track.is_live:
                    box_color = (0, 0, 200) # Deep red for spoof
                elif not dist_ok:
                    box_color = dist_color

                cv2.rectangle(annotated, (x, y), (x + fw, y + fh), box_color, 2)

                # Distance badge top-right of box
                dist_badge = f"{track.distance_cm}cm ({track.distance_status})"
                cv2.putText(
                    annotated,
                    dist_badge,
                    (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    dist_color,
                    1,
                    cv2.LINE_AA,
                )

                # Name tag at bottom of box
                label = f"{track.recognized_name}"
                if track.is_known:
                    label += f" ({track.confidence_percentage:.0f}%)"
                if not track.is_live:
                    label = "[SPOOF REJECTED]"

                # Filled banner
                banner_h = 24
                cv2.rectangle(
                    annotated,
                    (x, y + fh - banner_h),
                    (x + fw, y + fh),
                    box_color,
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x + 5, y + fh - 7),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Global Overlay Indicators:
        if draw_overlay and annotated is not None:
            # 1. Optimal Distance banner (Top-Left, from original final.py)
            cv2.putText(
                annotated,
                f"Optimal Distance: {self.config.optimal_distance_cm}cm",
                (20, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 2. Multiple Faces warning (from original final.py)
            if is_multiple:
                warning_text = "WARNING: Multiple Faces Detected!"
                cv2.rectangle(annotated, (20, 50), (450, 90), (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    warning_text,
                    (30, 80),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 3. System FPS & Status (Top-Right)
            now = time.time()
            fps = 1.0 / max(0.001, now - self.prev_frame_time)
            self.prev_frame_time = now
            cv2.putText(
                annotated,
                f"SmartAttend AI | {fps:.1f} FPS",
                (w - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

        end_time = time.time()
        proc_ms = (end_time - start_time) * 1000.0

        return FrameProcessingResult(
            frame_timestamp=start_time,
            total_faces_detected=total_faces,
            multiple_faces_detected=is_multiple,
            processed_faces=face_events,
            annotated_frame=annotated,
            processing_time_ms=round(proc_ms, 2),
            fps=round(1000.0 / max(1.0, proc_ms), 1),
        )
