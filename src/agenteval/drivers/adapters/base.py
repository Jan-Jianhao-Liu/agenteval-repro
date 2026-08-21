"""Agent 适配器抽象基类：隔离 τ³-bench 外部接口，新增业务领域仅扩展适配器。"""

from __future__ import annotations

import abc
import hashlib
from typing import Optional


class AgentAdapter(abc.ABC):
    """黑盒智能体统一接口：只收发文本，不读取内部状态。"""

    def __init__(self, enable_dialog_cache: bool = True):
        self._dialog_cache: dict[str, str] = {}
        self.enable_dialog_cache = enable_dialog_cache

    # ------------------------------------------------------------- 抽象接口

    @abc.abstractmethod
    async def start_session(self) -> None:
        """开启一条新会话（τ³-bench 会话 / Ollama 本地上下文）。"""

    @abc.abstractmethod
    async def _raw_send(self, user_utterance: str) -> str:
        """实际发送文本并返回智能体回复（子类实现）。"""

    @abc.abstractmethod
    async def reset(self) -> None:
        """一键重置会话（回到全新状态）。"""

    @abc.abstractmethod
    async def close(self) -> None:
        """释放资源。"""

    # ------------------------------------------------------------- 公共实现

    async def send(self, user_utterance: str, prev_context: str = "") -> str:
        """带会话级缓存的发送：相同(输入+前置上下文)直接命中缓存。

        缓存键 = hash(输入 + 上一条智能体回复摘要)，避免有状态会话中
        "继续/下一步" 之类输入误命中错误缓存。
        """
        key = self._cache_key(user_utterance, prev_context)
        if self.enable_dialog_cache and key in self._dialog_cache:
            return self._dialog_cache[key]
        resp = await self._raw_send(user_utterance)
        if self.enable_dialog_cache:
            self._dialog_cache[key] = resp
        return resp

    @staticmethod
    def _cache_key(utterance: str, context: str) -> str:
        raw = f"{utterance}:::{context[-200:]}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]
