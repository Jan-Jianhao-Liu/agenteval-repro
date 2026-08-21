"""全链路 Trace 追踪：全局 TraceID 串联，结构化 JSONL 日志，支持离线回放。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 全局 TraceID（contextvar，异步并发下自动隔离）
_trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    return f"trc_{uuid.uuid4().hex[:12]}"


def get_trace_id() -> str:
    return _trace_id_ctx.get()


def set_trace_id(tid: str) -> None:
    _trace_id_ctx.set(tid)


class TraceLogger:
    """结构化 Trace 记录器：固定字段 + 可扩展 extra，JSONL 追加写。"""

    _lock = threading.Lock()

    def __init__(self, log_dir: str | Path, save_input_output: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_input_output = save_input_output

    def log(
        self,
        module: str,
        level: str = "INFO",
        *,
        input: Any = None,
        output: Any = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        rec: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": get_trace_id() or new_trace_id(),
            "module": module,
            "level": level,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "error": error,
        }
        if self.save_input_output:
            rec["input"] = input if isinstance(input, str) else _json_safe(input)
            rec["output"] = output if isinstance(output, str) else _json_safe(output)
        if extra:
            rec["extra"] = extra
        with self._lock:
            with (self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def replay(self, trace_id: str) -> list[dict]:
        """离线回放：按 trace_id 抽取全链路记录（调试/定位 Bug 用）。"""
        out = []
        for p in sorted(self.log_dir.glob("*.jsonl")):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("trace_id") == trace_id:
                        out.append(rec)
        return out


def _json_safe(obj: Any) -> Any:
    """非 JSON 原生对象转可序列化表示（实体 dict 化）。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj


# 进程级默认 logger（惰性初始化，由 cli 注入目录）
_default_logger: Optional[TraceLogger] = None
_default_lock = threading.Lock()


def get_logger() -> TraceLogger:
    global _default_logger
    if _default_logger is None:
        with _default_lock:
            if _default_logger is None:
                _default_logger = TraceLogger(Path("data/trace"))
    return _default_logger


def configure(logger: TraceLogger) -> None:
    global _default_logger
    with _default_lock:
        _default_logger = logger
