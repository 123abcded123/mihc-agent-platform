
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

class LocalObjectStore:

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        name = os.path.basename(name or "file")
        name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
        return name[:180] or "file"

    def _path(self, object_key: str) -> Path:
        path = (self.root / object_key).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("非法对象路径")
        return path

    def put_bytes(self, project_id: str, file_id: str, file_name: str, content: bytes) -> dict:
        safe_name = self._safe_name(file_name)
        object_key = f"projects/{self._safe_name(project_id)}/{file_id}/{safe_name}"
        path = self._path(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "object_key": object_key,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def put_json(self, object_key: str, value: Any) -> str:
        path = self._path(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return object_key

    def put_text(self, object_key: str, value: str) -> str:
        path = self._path(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return object_key

    def read_bytes(self, object_key: str) -> bytes:
        return self._path(object_key).read_bytes()

    def read_json(self, object_key: str) -> Any:
        return json.loads(self._path(object_key).read_text(encoding="utf-8"))

    def exists(self, object_key: str) -> bool:
        return self._path(object_key).exists()
