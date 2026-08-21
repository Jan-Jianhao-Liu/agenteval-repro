"""Ollama 本地模拟 τ³-bench 航空域智能体（离线开发/冒烟测试用）。

通过本机 Ollama 原生 /api/chat 接口驱动 qwen3.5 扮演带业务规则的
航空退票客服，规则与论文场景对齐：身份校验、退票资格、二次确认、
非法参数拦截，为事件抽象/边界测试提供可复现的黑盒对话源。
"""

from __future__ import annotations

import httpx

from .base import AgentAdapter

AIRLINE_SYSTEM_PROMPT = """你是航空公司的智能客服助手，负责处理用户的机票预订、退票、改签、行李、值机等业务。请遵循以下业务规则：

1. 身份校验：涉及退票、改签、查看订单时，必须先要求用户提供订单号，并对姓名进行核对；
2. 退票资格：只有「起飞前 24 小时以上」且「票种为可退」的机票才能免费退票；起飞前 24 小时内退票需收 50% 手续费；特价票（不可退票种）一律不可退；
3. 二次确认：执行退票操作前，必须向用户二次确认退票信息和退款方式；
4. 非法输入拦截：订单号格式应为 13 位数字（如 999-1234567890）；日期格式应为 YYYY-MM-DD；航班号格式为 2 位字母+3 位数字；输入不合法时礼貌要求重新提供；
5. 退款方式：可退至原支付方式或账户余额，需用户确认；
6. 回答保持简洁、礼貌，一次只推进一个业务步骤，必要时主动引导用户补充缺失信息。

你只能基于上述规则服务航空业务，不做其他闲聊。"""


class OllamaAgent(AgentAdapter):
    """本地 Ollama 模拟智能体：支持多轮上下文、会话重置。"""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b",
        system_prompt: str = AIRLINE_SYSTEM_PROMPT,
        num_ctx: int = 8192,
        temperature: float = 0.7,
        enable_dialog_cache: bool = True,
    ):
        super().__init__(enable_dialog_cache=enable_dialog_cache)
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or AIRLINE_SYSTEM_PROMPT
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._client: httpx.AsyncClient | None = None
        self._messages: list[dict[str, str]] = []

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def start_session(self) -> None:
        self._messages = [{"role": "system", "content": self.system_prompt}]

    async def _raw_send(self, user_utterance: str) -> str:
        self._messages.append({"role": "user", "content": user_utterance})
        payload = {
            "model": self.model,
            "messages": self._messages,
            "stream": False,
            "think": False,  # 本机 Ollama 0.17.1 原生接口支持运行时关闭思考
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        resp = await self.client.post(f"{self.api_base}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content", "").strip()
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    async def reset(self) -> None:
        self._messages = []

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
