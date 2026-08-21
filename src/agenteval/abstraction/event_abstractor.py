"""事件抽象器（L2）：对话文本 → 结构化动作标签（DialogueEvent 补全）。"""

from __future__ import annotations

from typing import Optional

from ..domain import SessionTrace
from ..llm import LLMGateway
from .abstract_cache import AbstractCache, _fingerprint
from .label_mapper import LABEL_GLOSSARY, LabelMapper
from .prompter import event_abstractor_system, event_abstractor_user


class EventAbstractor:
    """将单轮对话抽象为标准化标签；semantic_key 用于 DFG 图节点合并。

    流程：LLM 输出原始标签 → LabelMapper 语义规范化（token 匹配 + bge-m3 相似度）
     → 落 DialogueEvent；semantic_key 恒等于规范化后的 user_action。
    接入会话级抽象缓存：同一探索轨迹在消融三模式间复用抽象结果（方案 3.2）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        cache: Optional[AbstractCache] = None,
        mapper: Optional[LabelMapper] = None,
        taxonomy: Optional[list[str]] = None,
    ):
        self.gateway = gateway
        self.cache = cache
        self.mapper = mapper
        self.taxonomy = taxonomy or []
        self._system = event_abstractor_system(self.taxonomy or None, LABEL_GLOSSARY)

    async def abstract_turn(
        self, user_utterance: str, agent_response: str
    ) -> dict[str, str]:
        """单轮抽象，返回 {user_action, agent_activity, semantic_key}。"""
        raw = await self.gateway.complete(
            module="event_abstractor",
            system_prompt=self._system,
            user_prompt=event_abstractor_user(user_utterance, agent_response),
        )
        user_action = _norm_tag(raw.get("user_action"))
        agent_activity = _norm_tag(raw.get("agent_activity"))
        if self.mapper is not None:
            # taxonomy 直命中则跳过映射（省 embedding + 更准）；否则走语义规范化
            if not (self.taxonomy and user_action in self.taxonomy):
                user_action = await self.mapper.map_label(user_action)
            if not (self.taxonomy and agent_activity in self.taxonomy):
                agent_activity = await self.mapper.map_label(agent_activity)
        # semantic_key 恒等于规范化后的 user_action：图节点 = 用户意图（同义话术自动合并）
        return {
            "user_action": user_action,
            "agent_activity": agent_activity,
            "semantic_key": user_action,
        }

    async def abstract_session(self, trace: SessionTrace) -> SessionTrace:
        """抽象整条会话；命中会话级缓存时直接复用，否则抽象并落盘。"""
        if self.cache is not None:
            fp = _fingerprint(trace)
            cached = self.cache.get(trace.session_id, fingerprint=fp)
            if cached is not None:
                return cached
            await self._abstract_in_place(trace)
            self.cache.put(trace, fingerprint=fp)
            return trace
        await self._abstract_in_place(trace)
        return trace

    async def _abstract_in_place(self, trace: SessionTrace) -> None:
        for event in trace.events:
            tags = await self.abstract_turn(event.user_utterance, event.agent_response)
            event.user_action = tags["user_action"]
            event.agent_activity = tags["agent_activity"]
            event.semantic_key = tags["semantic_key"]


def _norm_tag(v: Optional[str]) -> str:
    """标签规范化：去空白、转小写；非法时降级 unknown。"""
    if not v:
        return "unknown_action"
    v = str(v).strip().lower().replace(" ", "_")
    return v if v else "unknown_action"
