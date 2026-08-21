"""τ³-bench 真实接入适配器（黑盒）。

通过 tau2-bench（sierra-research/tau2-bench）程序化 API 与被测智能体对话：
- 域装配：tau2.domains.{domain}.environment.get_environment()（工具后端 + 业务策略）；
- 被测智能体：tau2.agent.llm_agent.LLMAgent（LiteLLM 模型名，如 deepseek/deepseek-v4-flash）；
- 交互：UserMessage → agent.generate_next_message → 若工具调用则 environment.get_response
  执行并回喂（MultiToolMessage），循环直至 agent 输出纯文本回复；
- 会话隔离：每会话全新 Environment + LLMAgent（工具数据库独立实例，互不污染）。

需要 tau2-bench 已安装（pip install -e <tau2-bench 目录>），API key 经 .env 配置
（LiteLLM 读取，如 DEEPSEEK_API_KEY）。
"""

from __future__ import annotations

import importlib
from typing import Optional

from .base import AgentAdapter

# 域 -> tau2 装配模块路径（banking_knowledge 需 knowledge extra，按需加入）
_DOMAIN_MODULES = {
    "airline": "tau2.domains.airline.environment",
    "retail": "tau2.domains.retail.environment",
    "telecom": "tau2.domains.telecom.environment",
    "banking_knowledge": "tau2.domains.banking_knowledge.environment",
}


def _build_env(domain: str, retrieval_variant: Optional[str] = None):
    mod = importlib.import_module(_DOMAIN_MODULES[domain])
    if domain == "banking_knowledge":
        # 知识检索域：默认 bm25（纯离线稀疏检索，无需 OpenAI/OpenRouter embedding API）
        return mod.get_environment(retrieval_variant=retrieval_variant or "bm25")
    return mod.get_environment()


class T3BenchAdapter(AgentAdapter):
    """τ³-bench 被测智能体黑盒适配器：真实工具后端 + 真实 LLM。"""

    def __init__(
        self,
        domain: str = "airline",
        model: str = "deepseek/deepseek-v4-flash",
        temperature: float = 0.0,
        max_tool_rounds: int = 20,
        retrieval_variant: Optional[str] = None,
        enable_dialog_cache: bool = True,
    ):
        super().__init__(enable_dialog_cache=enable_dialog_cache)
        if domain not in _DOMAIN_MODULES:
            raise ValueError(
                f"不支持的 τ³-bench 域: {domain}（可选 {list(_DOMAIN_MODULES)}；"
                f"banking_knowledge 需安装 knowledge extra）"
            )
        self.domain = domain
        self.model = model
        self.llm_args = {"temperature": temperature, "max_tokens": 4096}
        self.max_tool_rounds = max_tool_rounds
        self.retrieval_variant = retrieval_variant
        self.env = None
        self.agent = None
        self.state = None

    # ------------------------------------------------------------ 会话管理

    async def start_session(self) -> None:
        from tau2.agent.llm_agent import LLMAgent

        self.env = _build_env(self.domain, self.retrieval_variant)
        self.agent = LLMAgent(
            tools=self.env.get_tools(),
            domain_policy=self.env.get_policy(),
            llm=self.model,
            llm_args=self.llm_args,
        )
        self.state = self.agent.get_init_state()

    async def _raw_send(self, user_utterance: str) -> str:
        from tau2.data_model.message import MultiToolMessage, UserMessage

        msg = UserMessage.text(content=user_utterance)
        for _ in range(self.max_tool_rounds):
            assistant_msg, self.state = self.agent.generate_next_message(msg, self.state)
            if not assistant_msg.is_tool_call():
                text = (assistant_msg.content or "").strip()
                if text:
                    return text
                # 空文本继续循环（极少数情况）
                continue
            # 执行工具调用并回喂结果
            tool_messages = [self.env.get_response(tc) for tc in assistant_msg.tool_calls]
            msg = MultiToolMessage(role="tool", tool_messages=tool_messages)
        return "(agent 工具循环超限，无文本回复)"

    async def reset(self) -> None:
        if self.agent is not None:
            self.state = self.agent.get_init_state()
            self.env = _build_env(self.domain, self.retrieval_variant)
            self.agent.tools = self.env.get_tools()
            self.agent.domain_policy = self.env.get_policy()

    async def close(self) -> None:
        self.env = None
        self.agent = None
        self.state = None
