
from __future__ import annotations

import os
import json
import logging
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import pymongo
except ImportError:
    pymongo = None

class JSONFileStore:

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, collection: str) -> str:
        return os.path.join(self.base_dir, f"{collection}.json")

    def _read(self, collection: str) -> List[dict]:
        path = self._path(collection)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []

    def _write(self, collection: str, data: List[dict]) -> None:
        with open(self._path(collection), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def insert_one(self, collection: str, doc: Dict[str, Any]) -> str:
        with self._lock:
            data = self._read(collection)
            doc_id = doc.get("_id") or os.urandom(8).hex()
            doc["_id"] = doc_id
            data.append(doc)
            self._write(collection, data)
            return doc_id

    def find(self, collection: str, query: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[dict]:
        data = self._read(collection)
        query = query or {}
        out = []
        for doc in data:
            if all(doc.get(k) == v for k, v in query.items()):
                out.append(doc)
                if len(out) >= limit:
                    break
        return out

class MongoStore:

    def __init__(self, config):
        self.config = config
        self.backend = "json_file"
        self.client = None
        if config.mongo.url and pymongo:
            self.client = pymongo.MongoClient(config.mongo.url)
            self.db = self.client[config.mongo.db]
            self.backend = "mongodb"
        else:
            self.db = JSONFileStore(config.mongo.local_dir)
        logger.info("Mongo store backend: %s", self.backend)

    def add_badcase(self, *, query: str, answer: str, intent: str = "", trace_id: str = "",
                    feedback: str = "", expert_label: str = "", root_cause: str = "",
                    stage: str = "collected") -> str:
        doc = {
            "query": query, "answer": answer, "intent": intent, "trace_id": trace_id,
            "feedback": feedback, "expert_label": expert_label, "root_cause": root_cause,
            "stage": stage, "created_at": datetime.now().isoformat(),
        }
        return self.db.insert_one("badcases", doc)

    def list_badcases(self, stage: Optional[str] = None, limit: int = 100) -> List[dict]:
        query = {"stage": stage} if stage else None
        return self.db.find("badcases", query, limit)

    def save_eval_result(self, report: Dict[str, Any]) -> str:
        report["created_at"] = datetime.now().isoformat()
        return self.db.insert_one("eval_results", report)

    def list_eval_results(self, limit: int = 20) -> List[dict]:
        return self.db.find("eval_results", None, limit)

    def save_dataset(self, name: str, samples: List[dict]) -> str:
        return self.db.insert_one("eval_datasets", {"name": name, "samples": samples,
                                                    "count": len(samples),
                                                    "created_at": datetime.now().isoformat()})

    def load_dataset(self, name: str) -> List[dict]:
        rows = self.db.find("eval_datasets", {"name": name}, limit=1)
        return rows[0].get("samples", []) if rows else []
