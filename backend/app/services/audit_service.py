"""
Audit Logging and Notification Services for SmartAttend AI.
"""

import time
import uuid
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List

from backend.app.database.mongodb import db_manager
from backend.app.config import settings

logger = logging.getLogger("SmartAttend.Services")


class AuditService:
    @staticmethod
    async def log_action(
        action: str,
        entity_type: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_role: Optional[str] = None,
        entity_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ):
        """Records an audit log entry for security and regulatory compliance."""
        audit_col = db_manager.get_collection("audit_logs")
        entry = {
            "id": f"AUD-{uuid.uuid4().hex[:12]}",
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "user_id": user_id,
            "user_email": user_email,
            "user_role": user_role,
            "details": details or {},
            "ip_address": ip_address,
            "timestamp": time.time(),
        }
        await audit_col.insert_one(entry)
        logger.info("[AUDIT] %s | User: %s | Entity: %s (%s)", action, user_email, entity_type, entity_id)


class NotificationService:
    @staticmethod
    async def create_in_app_notification(
        user_id: str,
        title: str,
        message: str,
        category: str = "INFO",  # "INFO", "WARNING", "ATTENDANCE", "SECURITY"
        link: Optional[str] = None,
    ):
        """Creates an in-app notification in the database."""
        notif_col = db_manager.get_collection("notifications")
        doc = {
            "id": f"NOTIF-{uuid.uuid4().hex[:10]}",
            "user_id": user_id,
            "title": title,
            "message": message,
            "category": category,
            "link": link,
            "is_read": False,
            "created_at": time.time(),
        }
        await notif_col.insert_one(doc)

    @staticmethod
    def send_smtp_email(
        recipient_email: str,
        subject: str,
        body_text: str,
    ) -> bool:
        """
        Sends an email notification via SMTP (Preserving and enhancing legacy email alert functionality).
        """
        if not settings.SMTP_ENABLED or not settings.SMTP_PASSWORD:
            logger.info("[SIMULATED EMAIL] To: %s | Subject: %s", recipient_email, subject)
            return True

        try:
            msg = MIMEMultipart()
            msg["From"] = settings.SMTP_USER
            msg["To"] = recipient_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body_text, "plain"))

            server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_USER, recipient_email, msg.as_string())
            server.quit()
            logger.info("Sent email successfully to %s", recipient_email)
            return True
        except Exception as e:
            logger.error("Failed to send SMTP email to %s: %s", recipient_email, e)
            return False
