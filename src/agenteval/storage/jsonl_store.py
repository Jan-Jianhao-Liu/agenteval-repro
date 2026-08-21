"""JSONL 存储工具类：存储逻辑独立，可替换介质，上层业务无感知。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class JsonlStore:
    """追加写 JSONL 文件集合（按子目录/文件名隔离）。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def append(self, name: str, record: dict[str, Any]) -> None:
        """追加一条记录（自动序列化 Pydantic 实体）。"""
        rec = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        with self._path(name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def read_all(self, name: str) -> list[dict[str, Any]]:
        return list(self.iter(name))

    def iter(self, name: str) -> Iterator[dict[str, Any]]:
        p = self._path(name)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def file(self, name: str) -> Path:
        return self._path(name)
