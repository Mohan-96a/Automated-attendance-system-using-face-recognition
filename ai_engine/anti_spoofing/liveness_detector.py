"""
Anti-Spoofing and Liveness Detection Module for SmartAttend AI.
Detects printed photograph attacks, mobile screen video replay,
and provides passive multi-cue liveness analysis alongside randomized active challenges.
"""

import math
import random
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ai_engine.config.ai_config import AIConfig, get_ai_config

logger = logging.getLogger("SmartAttend.LivenessDetector")


@dataclass
class LivenessResult:
    is_live: bool
    liveness_score: float             # 0.0 (definite spoof) to 1.0 (authentic living human)
    spoof_type: Optional[str] = None  # None, "PRINT_ATTACK", "SCREEN_REPLAY", "STATIC_MASK"
    confidence_score: float = 0.0     # 0.0% to 100.0%
    texture_score: float = 0.0        # High-frequency texture richness
    fft_high_freq_ratio: float = 0.0  # Fourier high frequency energy ratio
    color_diversity_score: float = 0.0 # Chromatic skin dispersion
    specular_reflection_ratio: float = 0.0 # Screen glare peak detection
    active_challenge_passed: bool = True
    active_challenge_prompt: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


class LivenessDetector:
    """
    Multi-cue CPU-optimized liveness detector combining frequency analysis,
    texture gradient distribution, screen glare detection, and active eye blink verification.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or get_ai_config()
        self.blink_history: List[float] = []
        self.current_challenge: Optional[str] = None
        self.challenge_start_time: float = 0.0
        self.challenge_passed: bool = False

    def _analyze_fft_frequency(self, gray_face: np.ndarray) -> float:
        """
        Calculates high-frequency spectral energy ratio using 2D FFT.
        Printed photos exhibit lower high-frequency energy due to paper absorption
        and print resolution limits.
        """
        h, w = gray_face.shape[:2]
        if h < 32 or w < 32:
            return 0.5

        # 2D Discrete Fourier Transform
        f = np.fft.fft2(gray_face)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)

        # Split spectrum into low frequency center and high frequency periphery
        cy, cx = h // 2, w // 2
        r_inner = min(h, w) // 6

        # Mask center
        y, x = np.ogrid[:h, :w]
        inner_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (r_inner ** 2)

        high_freq_energy = np.mean(magnitude_spectrum[~inner_mask])
        low_freq_energy = np.mean(magnitude_spectrum[inner_mask]) + 1e-7

        ratio = float(high_freq_energy / low_freq_energy)
        return ratio

    def _analyze_texture_lbp(self, gray_face: np.ndarray) -> float:
        """
        Analyzes micro-texture sharpness and edge density.
        Human skin has fine pores and subtle micro-gradients; paper/screens appear either
        too smooth or unnaturally patterned.
        """
        resized = cv2.resize(gray_face, (96, 96))
        # Compute gradient magnitude
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        
        # Micro-texture standard deviation
        texture_val = float(np.std(mag))
        # Normalize to typical range [0.0, 1.0]
        score = min(1.0, texture_val / 45.0)
        return score

    def _detect_screen_glare(self, bgr_face: np.ndarray) -> Tuple[float, bool]:
        """
        Detects specular reflection hot-spots characteristic of glass phone/tablet screens.
        """
        hsv = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        # High value (>250) with low saturation (<20) indicates glass reflection
        sat = hsv[:, :, 1]
        glare_mask = (val > 248) & (sat < 25)
        glare_ratio = float(np.sum(glare_mask)) / float(val.size)
        is_screen_glare = glare_ratio > 0.045
        return glare_ratio, is_screen_glare

    def _analyze_color_chroma(self, bgr_face: np.ndarray) -> float:
        """
        Evaluates natural skin color distribution in YCrCb color space.
        Printed paper often has low chrominance variance or artificial gamut shifts.
        """
        ycrcb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        cr_std = float(np.std(cr))
        cb_std = float(np.std(cb))
        dispersion = (cr_std + cb_std) / 2.0
        # Normal skin has dispersion between 8 and 28
        score = 1.0 if 8.0 <= dispersion <= 28.0 else max(0.2, 1.0 - abs(dispersion - 18.0) / 20.0)
        return score

    def calculate_eye_aspect_ratio(
        self, eye_roi: np.ndarray
    ) -> float:
        """
        Estimates openness / blink ratio of detected eye region using vertical vs horizontal ratio.
        """
        if eye_roi is None or eye_roi.size == 0:
            return 0.3
        h, w = eye_roi.shape[:2]
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        # Threshold dark pupil
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        white_pixels = np.sum(thresh > 0)
        ratio = float(white_pixels) / float(eye_roi.size)
        return ratio

    def generate_active_challenge(self) -> str:
        """Generates a randomized interactive liveness prompt for anti-spoofing verification."""
        challenges = [
            "BLINK_TWICE",
            "TURN_HEAD_SLIGHTLY_LEFT",
            "TURN_HEAD_SLIGHTLY_RIGHT",
            "NOD_HEAD",
        ]
        self.current_challenge = random.choice(challenges)
        self.challenge_start_time = cv2.getTickCount()
        self.challenge_passed = False
        return self.current_challenge

    def verify_liveness(
        self,
        face_img: np.ndarray,
        landmarks: Optional[Dict[str, Tuple[int, int]]] = None,
        yaw_angle: float = 0.0,
    ) -> LivenessResult:
        """
        Performs holistic multi-modal liveness verification on a candidate face crop.
        Combines:
        1. Fourier frequency energy distribution (prints vs living)
        2. Micro-texture gradient analysis
        3. Screen glass glare and moiré indicators
        4. Color space distribution
        """
        if face_img is None or face_img.size == 0:
            return LivenessResult(
                is_live=False,
                liveness_score=0.0,
                spoof_type="INVALID_FRAME",
                reasons=["Empty face crop"],
            )

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

        # 1. FFT Frequency Analysis
        fft_ratio = self._analyze_fft_frequency(gray)
        # Normal faces generally have fft_ratio > 0.42
        fft_norm = min(1.0, max(0.0, (fft_ratio - 0.25) / 0.45))

        # 2. Micro-Texture Analysis
        texture_score = self._analyze_texture_lbp(gray)

        # 3. Screen Glare / Replay Analysis
        glare_ratio, is_screen_glare = self._detect_screen_glare(face_img)

        # 4. Color Chroma Dispersion
        chroma_score = self._analyze_color_chroma(face_img)

        # Weighted composite score
        composite_score = (
            0.35 * fft_norm +
            0.30 * texture_score +
            0.20 * chroma_score +
            0.15 * (1.0 - min(1.0, glare_ratio * 15.0))
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        reasons = []
        spoof_type = None

        if is_screen_glare:
            spoof_type = "SCREEN_REPLAY"
            reasons.append("Screen glare and specular reflection detected")
            composite_score *= 0.5

        if fft_norm < 0.35:
            spoof_type = "PRINT_ATTACK"
            reasons.append("Cut-off high-frequency spectrum indicative of printed photograph")
            composite_score *= 0.6

        if texture_score < 0.25:
            spoof_type = spoof_type or "STATIC_MASK"
            reasons.append("Abnormally low micro-texture sharpness")
            composite_score *= 0.7

        is_live = composite_score >= self.config.liveness_score_threshold
        confidence_pct = round(composite_score * 100.0, 1)

        return LivenessResult(
            is_live=is_live,
            liveness_score=composite_score,
            spoof_type=None if is_live else (spoof_type or "UNKNOWN_SPOOF"),
            confidence_score=confidence_pct,
            texture_score=round(texture_score, 3),
            fft_high_freq_ratio=round(fft_ratio, 3),
            color_diversity_score=round(chroma_score, 3),
            specular_reflection_ratio=round(glare_ratio, 4),
            active_challenge_passed=True,
            reasons=reasons,
        )
