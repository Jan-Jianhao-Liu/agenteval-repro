"""LLM 探索规划器（L1/L2 桥接）：基于历史轨迹生成下一条探索语句。

替代 SimpleExplorer 的固定轮转，用 explore_planner 标准化 Prompt 驱动：
- 输入：历史对话轨迹、已采集活动清单、当前工作流图摘要；
- 输出：explore_objective / user_utterance / risk_repeat；
- 失败兜底：回退 SimpleExplorer 轮转（不中断会话采集）。
"""

from __future__ import annotations

from typing import Optional

from ..abstraction.prompter import explore_planner_system, explore_planner_user
from ..domain import SessionTrace
from ..llm import LLMGateway
from .session import SimpleExplorer


class LLMExplorer:
    """基于 LLM 的探索规划器：覆盖更多业务路径，扩大 DFG 覆盖面。"""

    def __init__(self, gateway: LLMGateway, max_history_turns: int = 6):
        self.gateway = gateway
        self.max_history_turns = max_history_turns
        self._system = explore_planner_system()
        self._fallback = SimpleExplorer()

    async def __call__(self, trace: SessionTrace, objective: str, turn_idx: int) -> str:
        if turn_idx == 0:
            return objective
        history = _history_text(trace, self.max_history_turns)
        activities = sorted({e.user_utterance[:30] for e in trace.events})
        try:
            raw = await self.gateway.complete(
                module="explore_planner",
                system_prompt=self._system,
                user_prompt=explore_planner_user(history, activities),
            )
            utterance = str(raw.get("user_utterance", "")).strip()
            if utterance:
                return utterance
        except Exception:  # noqa: BLE001 LLM 失败回退轮转
            pass
        return await self._fallback(trace, objective, turn_idx)


def _history_text(trace: SessionTrace, max_turns: int) -> str:
    lines = []
    for e in trace.events[-max_turns:]:
        lines.append(f"U: {e.user_utterance[:100]}")
        lines.append(f"A: {e.agent_response[:150]}")
    return "\n".join(lines) or "（暂无历史）"
