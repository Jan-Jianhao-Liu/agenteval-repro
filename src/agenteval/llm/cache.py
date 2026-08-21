"""LLM 中间结果缓存：以 (模块, prompt 模板指纹, 输入文本) 为 key，JSONL 落盘。"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional


class LLMCache:
    """分层缓存：支持按实验模式分目录隔离，命中直接返回，避免重复 API 调用。"""

    _lock = threading.Lock()

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._mem: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(module: str, prompt_fp: str, user_input: str) -> str:
        raw = f"{module}::{prompt_fp}::{user_input}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _file(self, module: str) -> Path:
        return self.root / f"{module}.jsonl"

    def get(self, module: str, prompt_fp: str, user_input: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        key = self._key(module, prompt_fp, user_input)
        # 1) 内存索引
        hit = self._mem.get(key)
        if hit is not None:
            self._hits += 1
            return dict(hit)
        # 2) 磁盘扫描（首次进程内加载）
        with self._lock:
            f = self._file(module)
            if f.exists():
                for line in f.open("r", encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec["key"] == key:
                        self._mem[key] = rec["value"]
                        self._hits += 1
                        return dict(rec["value"])
        self._misses += 1
        return None

    def put(self, module: str, prompt_fp: str, user_input: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        key = self._key(module, prompt_fp, user_input)
        rec = {"key": key, "module": module, "value": value}
        self._mem[key] = value
        with self._lock:
            with self._file(module).open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}
