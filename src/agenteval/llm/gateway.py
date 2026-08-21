"""LLM 统一网关：Ollama 直连（httpx）+ 指数退避重试 + 熔断 + 分层缓存 + 纯 JSON 强解析。

所有模块的 LLM 调用必须经由本网关，禁止绕过。
统一保障：幂等 request_id、Trace 记录、JSON 修复重试、硬编码兜底。

传输层说明：默认走 Ollama 原生 /api/chat（支持 think:false，离线稳定）；
litellm 作为可选 provider 扩展点（见 _acompletion 的 provider 分支），
不引入重型 tokenizer 依赖，避免离线环境编码文件下载竞态。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, Callable, Optional

import httpx

from ..config import LLMConfig
from ..trace.logger import TraceLogger, get_logger
from .cache import LLMCache
from .circuit import CircuitBreaker

# ---- 各模块硬编码兜底输出（LLM 彻底不可用时的最后防线） ----
FALLBACKS: dict[str, dict[str, Any]] = {
    "explore_planner": {
        "explore_objective": "",
        "user_utterance": "",
        "risk_repeat": True,
    },
    "event_abstractor": {
        "user_action": "unknown_action",
        "agent_activity": "unknown_activity",
        "semantic_key": "unknown",
    },
    "boundary_scorer": {
        "boundary_potential": 0.0,
        "guard_type": "非法输入校验",
        "perturb_suggest": "",
    },
    "test_generator": {
        "disturb_utterance": "",
        "pass_criteria": "",
        "fail_criteria": "",
    },
    "judge": {
        "verdict": "inconclusive",
        "judge_reason": "LLM 不可用，由硬编码规则兜底判定",
        "fault_type": "none",
    },
}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """尽力提取纯 JSON：先整体解析，再剥围栏，再截取最外层花括号。"""
    t = text.strip()
    if not t:
        return None
    # 1) 直接解析
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    # 2) ```json 围栏
    m = _JSON_BLOCK_RE.search(t)
    if m:
        try:
            v = json.loads(m.group(1).strip())
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    # 3) 截取首个 { 到末个 }（容忍前后缀废话）
    start, end = t.find("{"), t.rfind("}")
    if 0 <= start < end:
        try:
            v = json.loads(t[start : end + 1])
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    return None


class LLMGateway:
    """全模块统一 LLM 调用入口（L1 层）。"""

    def __init__(
        self,
        cfg: LLMConfig,
        cache: Optional[LLMCache] = None,
        logger: Optional[TraceLogger] = None,
        num_ctx: int = 8192,
    ):
        self.cfg = cfg
        self.cache = cache or LLMCache("data/cache")
        self.logger = logger or get_logger()
        self.circuit = CircuitBreaker(
            fail_threshold=cfg.circuit_fail_threshold,
            cooldown_sec=cfg.circuit_cooldown_sec,
        )
        self._model_key = f"{cfg.provider}/{cfg.model}"
        self._num_ctx = num_ctx
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ 主入口

    async def complete(
        self,
        module: str,
        system_prompt: str,
        user_prompt: str,
        fallback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """统一调用：缓存 → 熔断 → LLM → JSON 解析/修复 → 兜底。

        Args:
            module: 模块标识（event_abstractor / explore_planner / ...）
            system_prompt: 系统提示（含强制 JSON 约束）
            user_prompt: 用户输入
            fallback: 兜底 JSON（缺省取 FALLBACKS[module]）
        Returns:
            解析后的 dict（LLM 输出或兜底）
        """
        fb = fallback or FALLBACKS.get(module, {})
        prompt_fp = _fingerprint(system_prompt)

        # 1) 缓存命中
        if self.cache.enabled:
            cached = self.cache.get(module, prompt_fp, user_prompt)
            if cached is not None:
                self.logger.log(module, "INFO", input=user_prompt, output=cached, extra={"hit": "cache"})
                return cached

        # 2) 熔断开启 → 直接兜底
        if self.circuit.is_open:
            self.logger.log(module, "WARN", input=user_prompt, output=fb,
                            error="circuit open, fallback", extra={"circuit": self.circuit.state})
            return dict(fb)

        request_id = f"req_{uuid.uuid4().hex[:10]}"
        t0 = time.monotonic()
        try:
            raw = await self._call_with_retry(module, system_prompt, user_prompt)
            parsed = _extract_json(raw)
            if parsed is None:
                parsed = await self._repair_json(module, system_prompt, user_prompt, raw)
            if parsed is None:
                self.logger.log(module, "WARN", input=user_prompt, output=raw,
                                error="json parse failed after repair, fallback", extra={"request_id": request_id})
                return dict(fb)
            self.circuit.record_success()
            latency = (time.monotonic() - t0) * 1000
            self.logger.log(module, "INFO", input=user_prompt, output=parsed,
                            latency_ms=latency, extra={"request_id": request_id, "raw_len": len(raw)})
            self.cache.put(module, prompt_fp, user_prompt, parsed)
            return parsed
        except Exception as e:  # noqa: BLE001 网关兜底：任何异常不向业务层抛出
            self.circuit.record_failure()
            latency = (time.monotonic() - t0) * 1000
            self.logger.log(module, "ERROR", input=user_prompt, output=fb,
                            latency_ms=latency, error=f"{type(e).__name__}: {e}",
                            extra={"request_id": request_id, "circuit": self.circuit.state})
            return dict(fb)

    # ------------------------------------------------------------------ 内部实现

    async def _call_with_retry(self, module: str, system_prompt: str, user_prompt: str) -> str:
        """指数退避重试（1s/3s/5s...），超时/限流/网络类异常均重试。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_err: Exception = RuntimeError("unknown")
        backoff = list(self.cfg.retry_backoff) + [5.0] * 3  # 至少 3 次兜底退避
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return await self._acompletion(messages)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.cfg.max_retries:
                    wait = backoff[min(attempt, len(backoff) - 1)]
                    self.logger.log(module, "WARN", input=user_prompt, error=f"retry in {wait}s: {e}",
                                    extra={"attempt": attempt + 1})
                    await asyncio.sleep(wait)
        raise last_err

    async def _acompletion(self, messages: list[dict[str, str]]) -> str:
        """按 provider 路由到实际传输实现（默认 Ollama 原生 /api/chat）。

        扩展点：新增 provider 时仅在此处加分支，重试/熔断/缓存/兜底全部复用。
        """
        if self.cfg.provider in {"ollama", "ollama_chat"}:
            return await self._ollama_chat(messages)
        if self.cfg.provider == "openai_compat":
            return await self._openai_chat(messages)
        raise ValueError(f"不支持的 LLM provider: {self.cfg.provider}")

    async def _ollama_chat(self, messages: list[dict[str, str]]) -> str:
        """Ollama 原生 /api/chat：支持 think:false（原生 Qwen3.5 免思考）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.cfg.timeout_sec)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self._num_ctx, "temperature": self.cfg.temperature},
        }
        resp = await self._client.post(f"{self.cfg.api_base.rstrip('/')}/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content")
        if not content:
            raise ValueError("LLM 返回空内容")
        return str(content)

    async def _openai_chat(self, messages: list[dict[str, str]]) -> str:
        """OpenAI 兼容 /v1/chat/completions：DeepSeek / Kimi / Qwen / OpenAI 等真实大模型。"""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.cfg.api_key:
                headers["Authorization"] = f"Bearer {self.cfg.api_key}"
            self._client = httpx.AsyncClient(timeout=self.cfg.timeout_sec, headers=headers)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "temperature": self.cfg.temperature,
            "max_tokens": 2048,
        }
        base = (self.cfg.base_url or self.cfg.api_base).rstrip("/")
        resp = await self._client.post(f"{base}/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("LLM 返回空内容")
        return str(content)

    async def _repair_json(
        self, module: str, system_prompt: str, user_prompt: str, raw: str
    ) -> Optional[dict[str, Any]]:
        """JSON 解析失败：追加修复指令重试（默认 2 次）。"""
        repair_instruction = (
            "你上一轮输出不是合法 JSON。请仅输出一个合法 JSON 对象，"
            "不要 Markdown、不要解释、不要多余字符。上一轮输出如下：\n"
            f"{raw[:2000]}"
        )
        for i in range(self.cfg.json_fix_retries):
            try:
                fixed = await self._acompletion([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": repair_instruction},
                ])
            except Exception as e:  # noqa: BLE001
                self.logger.log(module, "WARN", error=f"repair attempt {i + 1} failed: {e}")
                continue
            parsed = _extract_json(fixed)
            if parsed is not None:
                self.logger.log(module, "INFO", output=parsed, extra={"repair_round": i + 1})
                return parsed
        return None


def _fingerprint(system_prompt: str) -> str:
    """Prompt 模板指纹：模板变动即失效缓存。"""
    import hashlib
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
