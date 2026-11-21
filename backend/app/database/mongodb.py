"""
Database Management Layer for SmartAttend AI.
Provides unified async document interface supporting both MongoDB (Motor)
and resilient local JSON document store when MongoDB is offline.
"""

import os
import json
import uuid
import time
import copy
import logging
from typing import Dict, List, Optional, Any

from backend.app.config import settings

logger = logging.getLogger("SmartAttend.Database")


class JSONCollection:
    """Async-compatible in-memory and file-backed JSON collection fallback."""

    def __init__(self, name: str, db_manager: "DatabaseManager"):
        self.name = name
        self.db_manager = db_manager

    def _get_items(self) -> List[Dict[str, Any]]:
        return self.db_manager.json_data.setdefault(self.name, [])

    async def find(self, query: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        query = query or {}
        results = []
        for doc in self._get_items():
            match = True
            for k, v in query.items():
                if k == "$or" and isinstance(v, list):
                    or_match = False
                    for cond in v:
                        if all(doc.get(ck) == cv for ck, cv in cond.items()):
                            or_match = True
                            break
                    if not or_match:
                        match = False
                        break
                elif doc.get(k) != v:
                    match = False
                    break
            if match:
                results.append(copy.deepcopy(doc))
                if limit > 0 and len(results) >= limit:
                    break
        return results

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        res = await self.find(query, limit=1)
        return res[0] if res else None

    async def insert_one(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc_copy = copy.deepcopy(doc)
        if "id" not in doc_copy and "_id" not in doc_copy:
            doc_copy["id"] = str(uuid.uuid4())
        elif "_id" in doc_copy and "id" not in doc_copy:
            doc_copy["id"] = str(doc_copy["_id"])
        
        doc_copy.setdefault("created_at", time.time())
        doc_copy.setdefault("updated_at", time.time())
        
        self._get_items().append(doc_copy)
        self.db_manager.save_json_db()
        return doc_copy

    async def insert_many(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inserted = []
        for doc in docs:
            res = await self.insert_one(doc)
            inserted.append(res)
        return inserted

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        items = self._get_items()
        set_fields = update.get("$set", update)
        modified = 0
        for doc in items:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                for sk, sv in set_fields.items():
                    doc[sk] = sv
                doc["updated_at"] = time.time()
                modified += 1
                break
        if modified > 0:
            self.db_manager.save_json_db()
        return modified

    async def delete_one(self, query: Dict[str, Any]) -> int:
        items = self._get_items()
        initial_len = len(items)
        for i, doc in enumerate(items):
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                del items[i]
                break
        if len(items) < initial_len:
            self.db_manager.save_json_db()
            return 1
        return 0

    async def count_documents(self, query: Optional[Dict[str, Any]] = None) -> int:
        res = await self.find(query)
        return len(res)


class DatabaseManager:
    """Manages connections to MongoDB and fallback storage."""

    COLLECTIONS = [
        "users",
        "students",
        "faculty",
        "departments",
        "subjects",
        "classrooms",
        "timetables",
        "face_embeddings",
        "attendance_sessions",
        "attendance_records",
        "notifications",
        "correction_requests",
        "audit_logs",
        "system_settings",
    ]

    def __init__(self):
        self.is_connected = False
        self.client = None  # Motor AsyncIOMotorClient — lazy-imported when needed
        self.db = None
        self.json_data: Dict[str, List[Dict[str, Any]]] = {}
        self.fallback_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "db.json"
        )
        self.collections: Dict[str, Any] = {}

    def load_json_db(self):
        """Loads fallback JSON data from disk."""
        os.makedirs(os.path.dirname(self.fallback_file), exist_ok=True)
        if os.path.exists(self.fallback_file):
            try:
                with open(self.fallback_file, "r", encoding="utf-8") as f:
                    self.json_data = json.load(f)
                logger.info("Loaded local JSON database from %s", self.fallback_file)
            except Exception as e:
                logger.warning("Could not read %s, initializing empty: %s", self.fallback_file, e)
                self.json_data = {}
        for coll in self.COLLECTIONS:
            self.json_data.setdefault(coll, [])

    def save_json_db(self):
        """Saves JSON data back to disk."""
        try:
            with open(self.fallback_file, "w", encoding="utf-8") as f:
                json.dump(self.json_data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to write JSON db: %s", e)

    async def connect(self):
        """Attempts connection to MongoDB; falls back to JSON document store gracefully."""
        print("[DB] connect() entered")
        self.load_json_db()
        print("[DB] load_json_db() done")
        mongodb_uri = os.getenv("MONGODB_URI")
        
        # If no explicit MONGODB_URI set or FORCE_JSON_DB is enabled, use zero-latency local JSON engine
        if not mongodb_uri or os.getenv("FORCE_JSON_DB", "true").lower() == "true":
            self.is_connected = False
            for col_name in self.COLLECTIONS:
                self.collections[col_name] = JSONCollection(col_name, self)
            logger.info("Activated high-speed resilient local JSON database engine.")
            print("[DB] Using JSON engine done")
            return

        try:
            import motor.motor_asyncio  # lazy import — only when MongoDB is actually needed
            from pymongo.errors import ConnectionFailure  # noqa: F401
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.MONGODB_URI, serverSelectionTimeoutMS=1000
            )
            # Ping
            await self.client.admin.command("ping")
            self.db = self.client[settings.DATABASE_NAME]
            self.is_connected = True
            for col_name in self.COLLECTIONS:
                self.collections[col_name] = self.db[col_name]
            logger.info("Connected to MongoDB at %s (DB: %s)", settings.MONGODB_URI, settings.DATABASE_NAME)
        except Exception as e:
            logger.warning(
                "MongoDB not reachable (%s). Activating resilient local JSON fallback engine.", e
            )
            self.is_connected = False
            for col_name in self.COLLECTIONS:
                self.collections[col_name] = JSONCollection(col_name, self)

    def get_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = JSONCollection(name, self)
        return self.collections[name]


db_manager = DatabaseManager()


def get_db(col_name: str):
    return db_manager.get_collection(col_name)
