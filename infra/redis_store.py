
from __future__ import annotations

import json
import time
import logging
import uuid
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

try:
    import fakeredis
except ImportError:
    fakeredis = None

class SessionStore:

    def __init__(self, config):
        self.config = config
        self.ttl = config.redis.session_ttl
        self.backend = "fakeredis"
        if not config.redis.use_fake and redis_lib:
            self.client = redis_lib.Redis.from_url(config.redis.url, decode_responses=True)
            self.backend = "redis"
        else:
            self.client = fakeredis.FakeRedis(decode_responses=True)
        logger.info("Session store backend: %s", self.backend)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"sess:{session_id}"

    def create_session(self, tenant_id: str = "mihc", user_id: str = "anonymous") -> str:
        session_id = uuid.uuid4().hex
        payload = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "messages": [],
            "intent": "",
            "task_state": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self.client.set(self._key(session_id), json.dumps(payload, ensure_ascii=False), ex=self.ttl)
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    def exists(self, session_id: str) -> bool:
        return self.client.exists(self._key(session_id)) > 0

    def save(self, payload: Dict[str, Any]) -> None:
        payload["updated_at"] = time.time()
        self.client.set(self._key(payload["session_id"]), json.dumps(payload, ensure_ascii=False), ex=self.ttl)

    def append_message(self, session_id: str, role: str, content: str, max_history: int = 20) -> Dict[str, Any]:
        payload = self.get(session_id) or self._payload_for(session_id)
        payload["messages"].append({"role": role, "content": content, "ts": time.time()})
        if len(payload["messages"]) > max_history:
            payload["messages"] = payload["messages"][-max_history:]
        self.save(payload)
        return payload

    def set_intent(self, session_id: str, intent: str) -> None:
        payload = self.get(session_id)
        if payload:
            payload["intent"] = intent
            self.save(payload)

    def set_task_state(self, session_id: str, task_state: Dict[str, Any]) -> None:
        payload = self.get(session_id)
        if payload:
            payload["task_state"] = task_state
            self.save(payload)

    def clear(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))

    def history(self, session_id: str) -> List[Dict[str, Any]]:
        payload = self.get(session_id)
        return payload["messages"] if payload else []

    def _payload_for(self, session_id: str) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "tenant_id": self.config.platform.tenant_id,
            "user_id": "anonymous",
            "messages": [],
            "intent": "",
            "task_state": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
