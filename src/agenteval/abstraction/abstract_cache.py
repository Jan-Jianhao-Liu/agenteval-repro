"""会话级抽象缓存：按 session_id + 对话内容指纹落盘抽象结果。

动机：消融实验三模式共享同一套探索会话轨迹（方案 3.2），事件抽象结果
完全可复用。按「对话内容指纹」判定是否命中——指纹只依赖原始用户语句与
智能体回复，与 LLM 抽象输出无关，故三模式间稳定命中。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ..domain import SessionTrace


def _fingerprint(trace: SessionTrace) -> str:
    """对话内容指纹：逐轮 user+agent 文本 sha256（不含 LLM 抽象结果）。"""
    h = hashlib.sha256()
    for e in trace.events:
        h.update(e.user_utterance.encode("utf-8"))
        h.update(b"||")
        h.update(e.agent_response.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


class AbstractCache:
    """落盘目录 data/{env}/abstracted/{session_id}.jsonl，内容 {fingerprint, trace}。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def get(self, session_id: str, fingerprint: Optional[str] = None) -> Optional[SessionTrace]:
        """命中条件：文件存在且指纹一致。"""
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if fingerprint is not None and rec.get("fingerprint") != fingerprint:
            return None
        return SessionTrace.model_validate(rec["trace"])

    def put(self, trace: SessionTrace, fingerprint: Optional[str] = None) -> None:
        rec = {
            "fingerprint": fingerprint or _fingerprint(trace),
            "trace": trace.model_dump(mode="json"),
        }
        self._path(trace.session_id).write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
