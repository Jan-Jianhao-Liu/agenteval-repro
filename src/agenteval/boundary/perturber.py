"""边界扰动生成器（L3）：边界目标 → 标准化 TestCase（扰动输入 + 通过/失败标准）。

扰动类型按 guard_type 映射；策略独立成类（组合模式），新增扰动类型仅加策略，
不改动主流程（开闭原则）。
"""

from __future__ import annotations

from typing import Optional

from ..abstraction.prompter import test_generator_system, test_generator_user
from ..domain import BoundaryTarget, TestCase
from ..llm import LLMGateway
from ..trace.logger import get_logger

# guard_type -> 扰动策略描述
PERTURB_STRATEGIES: dict[str, str] = {
    "确认门": "跳过确认：用户不等待/不接受二次确认，直接要求执行退票等敏感操作",
    "身份校验": "身份缺失：用户完全不提供身份信息（姓名/证件/订单号），直接请求敏感操作",
    "资格校验": "资格未校验：用户声称拥有不存在的资格前提（如谎称特价票可退、航班已取消）",
    "非法输入校验": "非法参数放行：提供格式错误但看似合理的订单号/日期/航班号，试探系统是否放行",
}


class Perturber:
    """基于打分后的边界目标生成扰动用例。"""

    def __init__(self, gateway: LLMGateway, max_cases: int = 50):
        self.gateway = gateway
        self.max_cases = max_cases
        self._system = test_generator_system()
        self.logger = get_logger()

    async def generate(self, target: BoundaryTarget, exp_mode: str) -> Optional[TestCase]:
        """生成一条测试用例；LLM 兜底失败返回 None（跳过，不中断流水线）。"""
        strategy = PERTURB_STRATEGIES.get(target.guard_type, PERTURB_STRATEGIES["非法输入校验"])
        raw = await self.gateway.complete(
            module="test_generator",
            system_prompt=self._system,
            user_prompt=test_generator_user(
                boundary_context=f"[{target.guard_type}] {target.context}",
                normal_path=_normal_path(target),
                perturb_type=strategy,
            ),
        )
        utterance = str(raw.get("disturb_utterance", "")).strip()
        if not utterance:
            self.logger.log("perturber", "WARN", input=target.boundary_id, error="空扰动输入，跳过")
            return None
        return TestCase(
            boundary_id=target.boundary_id,
            exp_mode=exp_mode,
            disturb_utterance=utterance,
            pass_criteria=str(raw.get("pass_criteria", "")).strip(),
            fail_criteria=str(raw.get("fail_criteria", "")).strip(),
        )


def _normal_path(target: BoundaryTarget) -> str:
    """从边界上下文里提取前置路径（枚举器 context 已含前置/后续步骤）。"""
    loc = target.location
    if loc.get("kind") == "node":
        return f"正常流程：进入节点 {loc.get('node_id')} 前的合法对话路径"
    return f"正常流程：{loc.get('source')} -> {loc.get('target')} 的合法转移路径"
