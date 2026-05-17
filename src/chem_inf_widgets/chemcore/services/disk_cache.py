from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CachePolicy:
    enabled: bool = True
    ttl_s: int = 24 * 3600  # 24h


class DiskCache:
    """
    Very small JSON disk cache (key -> {ts, value}), TTL-based.
    Safe for simple API responses and parsed lists.
    """

    def __init__(self, namespace: str = "chembl", base_dir: Optional[str] = None) -> None:
        if base_dir is None:
            base_dir = os.path.join(Path.home(), ".cache", "chem_inf_widgets")
        self.root = Path(base_dir) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{h}.json"

    def get(self, key: str, ttl_s: int) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ts = float(payload.get("ts", 0.0))
            if (time.time() - ts) > float(ttl_s):
                return None
            return payload.get("value", None)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"ts": time.time(), "value": value}), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            # cache must never break app
            return
