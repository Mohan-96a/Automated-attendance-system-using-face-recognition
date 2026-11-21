"""
Face Recognition and Biometric Template Management for SmartAttend AI.
Performs cosine-similarity matching, confidence scoring, unknown face detection,
biometric encryption/decryption, and duplicate recognition prevention.
"""

import os
import time
import base64
import logging
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from cryptography.fernet import Fernet
from ai_engine.config.ai_config import AIConfig, get_ai_config
from ai_engine.recognition.embedding_generator import EmbeddingGenerator

logger = logging.getLogger("SmartAttend.FaceRecognizer")


@dataclass
class RecognitionResult:
    matched: bool
    student_id: Optional[str] = None
    name: Optional[str] = None
    similarity_score: float = 0.0      # Cosine similarity (-1.0 to 1.0)
    confidence_percentage: float = 0.0  # 0.0% to 100.0%
    is_unknown: bool = False
    is_duplicate_cooldown: bool = False
    cooldown_remaining_sec: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class RegisteredPerson:
    student_id: str
    name: str
    encrypted_embedding: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BiometricSecurity:
    """Handles military-grade AES encryption of biometric facial templates."""

    def __init__(self, secret_key: str):
        # Derive 32-byte urlsafe base64 key from secret
        hashed = hashlib.sha256(secret_key.encode("utf-8")).digest()
        b64_key = base64.urlsafe_b64encode(hashed)
        self.cipher = Fernet(b64_key)

    def encrypt_embedding(self, embedding: np.ndarray) -> str:
        """Encrypts 512-d float32 array into secure string."""
        raw_bytes = embedding.astype(np.float32).tobytes()
        encrypted = self.cipher.encrypt(raw_bytes)
        return encrypted.decode("ascii")

    def decrypt_embedding(self, encrypted_str: str) -> np.ndarray:
        """Decrypts secure string back to 512-d float32 array."""
        decrypted = self.cipher.decrypt(encrypted_str.encode("ascii"))
        return np.frombuffer(decrypted, dtype=np.float32)


class FaceRecognizer:
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or get_ai_config()
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.security = BiometricSecurity(self.config.biometric_secret_key)
        
        # In-memory fast cache of decrypted normalized embeddings:
        # student_id -> {"name": str, "embedding": np.ndarray, "metadata": dict}
        self.known_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Last recognized timestamps for cooldown / duplicate prevention:
        # student_id -> last_timestamp
        self.last_recognized_times: Dict[str, float] = {}

    def register_face(
        self,
        student_id: str,
        name: str,
        aligned_face: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegisteredPerson:
        """
        Extracts embedding from aligned face, encrypts it, and adds to registry.
        """
        embedding = self.embedding_generator.generate_embedding(aligned_face)
        encrypted_str = self.security.encrypt_embedding(embedding)

        self.known_profiles[student_id] = {
            "name": name,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        logger.info("Registered face for %s (%s). Total known: %d", name, student_id, len(self.known_profiles))
        return RegisteredPerson(
            student_id=student_id,
            name=name,
            encrypted_embedding=encrypted_str,
            metadata=metadata or {},
        )

    def load_encrypted_profile(
        self,
        student_id: str,
        name: str,
        encrypted_embedding: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Loads and decrypts a biometric template from database/storage."""
        try:
            embedding = self.security.decrypt_embedding(encrypted_embedding)
            # Re-normalize to guard against float precision artifacts
            norm = np.linalg.norm(embedding)
            if norm > 1e-7:
                embedding = embedding / norm

            self.known_profiles[student_id] = {
                "name": name,
                "embedding": embedding,
                "metadata": metadata or {},
            }
            logger.debug("Loaded encrypted profile for %s (%s)", name, student_id)
        except Exception as e:
            logger.error("Failed to decrypt biometric profile for %s: %s", student_id, e)

    def unregister_face(self, student_id: str) -> bool:
        """Removes a student's biometric data from memory (GDPR deletion compliance)."""
        if student_id in self.known_profiles:
            del self.known_profiles[student_id]
            self.last_recognized_times.pop(student_id, None)
            logger.info("Deleted biometric template for student_id: %s", student_id)
            return True
        return False

    def clear_registry(self):
        """Wipes all biometric profiles from memory."""
        self.known_profiles.clear()
        self.last_recognized_times.clear()

    def recognize(
        self, aligned_face: np.ndarray, enforce_cooldown: bool = True
    ) -> RecognitionResult:
        """
        Matches an aligned face against all registered templates using Cosine Similarity.
        Handles thresholding, unknown face detection, and duplicate cooldown.
        """
        if aligned_face is None or aligned_face.size == 0 or len(self.known_profiles) == 0:
            return RecognitionResult(
                matched=False,
                is_unknown=True,
                name=self.config.unknown_label,
                similarity_score=0.0,
                confidence_percentage=0.0,
            )

        query_embedding = self.embedding_generator.generate_embedding(aligned_face)

        best_score = -1.0
        best_student_id = None
        best_name = None

        # Cosine similarity for unit vectors: dot product
        for s_id, profile in self.known_profiles.items():
            known_emb = profile["embedding"]
            sim = float(np.dot(query_embedding, known_emb))
            if sim > best_score:
                best_score = sim
                best_student_id = s_id
                best_name = profile["name"]

        # Confidence percentage mapping:
        # e.g. similarity ranges from -1 to 1; threshold is 0.62.
        # We map threshold (0.62) -> 65%, 1.0 -> 100%, below threshold -> below 50%
        thresh = self.config.recognition_threshold
        if best_score >= thresh:
            # Linear scaling between threshold and 1.0 to 65% - 99.9%
            fraction = (best_score - thresh) / max(0.01, 1.0 - thresh)
            confidence_pct = round(65.0 + (fraction * 34.9), 1)
        else:
            fraction = max(0.0, best_score / max(0.01, thresh))
            confidence_pct = round(fraction * 50.0, 1)

        is_match = best_score >= thresh
        now = time.time()

        if is_match and best_student_id:
            # Check duplicate recognition cooldown
            is_duplicate = False
            cooldown_rem = 0.0
            if enforce_cooldown and best_student_id in self.last_recognized_times:
                elapsed = now - self.last_recognized_times[best_student_id]
                if elapsed < self.config.recognition_cooldown_seconds:
                    is_duplicate = True
                    cooldown_rem = round(self.config.recognition_cooldown_seconds - elapsed, 1)

            if not is_duplicate:
                self.last_recognized_times[best_student_id] = now

            return RecognitionResult(
                matched=True,
                student_id=best_student_id,
                name=best_name,
                similarity_score=best_score,
                confidence_percentage=confidence_pct,
                is_unknown=False,
                is_duplicate_cooldown=is_duplicate,
                cooldown_remaining_sec=cooldown_rem,
                embedding=query_embedding,
            )
        else:
            return RecognitionResult(
                matched=False,
                student_id=None,
                name=self.config.unknown_label,
                similarity_score=best_score,
                confidence_percentage=confidence_pct,
                is_unknown=True,
                is_duplicate_cooldown=False,
                embedding=query_embedding,
            )
