"""边界打分器（L3）：LLM 评估边界潜力，筛选高潜力目标。

- 输入：图元素描述 + 对话上下文 + 已测试边界清单；
- 输出：boundary_potential(0~1) / guard_type(四选一) / perturb_suggest；
- guard_type 中文关键词校准；potential < threshold 的候选被过滤。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..abstraction.prompter import boundary_scorer_system, boundary_scorer_user
from ..domain import BoundaryTarget
from ..llm import LLMGateway
from ..trace.logger import get_logger
from .enumerator import GUARD_TYPES

_GUARD_CANON = {"确认门": "确认门", "身份校验": "身份校验", "资格校验": "资格校验", "非法输入校验": "非法输入校验"}


def _canon_guard(v: str) -> str:
    """LLM 输出校准：中文关键词 → 四类规范值。"""
    for kw, canon in _GUARD_CANON.items():
        if kw in v:
            return canon
    return "非法输入校验"


class BoundaryScorer:
    """对候选边界批量打分（并发受控），返回潜力达标的 BoundaryTarget。"""

    def __init__(self, gateway: LLMGateway, threshold: float = 0.5, concurrency: int = 4):
        self.gateway = gateway
        self.threshold = threshold
        self.concurrency = concurrency
        self._system = boundary_scorer_system()
        self.logger = get_logger()

    async def score(
        self, targets: list[BoundaryTarget], tested: Optional[list[str]] = None
    ) -> list[BoundaryTarget]:
        """返回打分后（potential 已更新、guard_type 已校准、过滤低潜力）的候选。"""
        tested = tested or []
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(t: BoundaryTarget) -> Optional[BoundaryTarget]:
            async with sem:
                raw = await self.gateway.complete(
                    module="boundary_scorer",
                    system_prompt=self._system,
                    user_prompt=boundary_scorer_user(
                        _target_desc(t), t.context, tested
                    ),
                )
            try:
                potential = max(0.0, min(1.0, float(raw.get("boundary_potential", 0.0))))
            except (TypeError, ValueError):
                potential = 0.0
            if potential < self.threshold:
                return None
            t.potential = potential
            t.guard_type = _canon_guard(str(raw.get("guard_type", "非法输入校验")))
            return t

        results = await asyncio.gather(*[_one(t) for t in targets])
        scored = [t for t in results if t is not None]
        scored.sort(key=lambda t: -t.potential)
        self.logger.log(
            "boundary_scorer", "INFO",
            input={"candidates": len(targets), "kept": len(scored), "threshold": self.threshold},
        )
        return scored


def _target_desc(t: BoundaryTarget) -> str:
    loc = t.location
    if loc.get("kind") == "node":
        return f"节点 {loc.get('node_id')}"
    return f"边 {loc.get('source')} -> {loc.get('target')}"
