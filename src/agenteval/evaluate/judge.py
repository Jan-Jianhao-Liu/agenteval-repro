"""测试裁判 Judge（L4）：判定边界测试结果。

顶层接口 JudgeProtocol 支持替换实现（纯规则 / 纯 LLM / 混合）；
LLMJudge 为 LLM 实现：verdict(pass/fail/inconclusive) + judge_reason + fault_type。
"""

from __future__ import annotations

from typing import Optional, Protocol

from ..abstraction.prompter import judge_system, judge_user
from ..domain import SessionTrace, TestCase
from ..llm import LLMGateway

VERDICTS = {"pass", "fail", "inconclusive"}

# judge prompt 输出中文 fault_type -> 规范英文枚举
_FAULT_CANON = {
    "跳过确认": "skip_confirmation",
    "身份缺失": "missing_identity",
    "资格未校验": "missing_eligibility",
    "非法参数放行": "illegal_param_passed",
    "无": "none",
}


def _canon_verdict(v: str) -> str:
    v = str(v).strip().lower()
    return v if v in VERDICTS else "inconclusive"


def _canon_fault(v: str) -> str:
    for kw, canon in _FAULT_CANON.items():
        if kw in str(v):
            return canon
    return "none"


class JudgeProtocol(Protocol):
    async def judge(self, trace: SessionTrace, case: TestCase) -> TestCase: ...


class LLMJudge:
    """LLM 裁判：完整测试会话 + 用例预期 → 判定。"""

    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway
        self._system = judge_system()

    async def judge(self, trace: SessionTrace, case: TestCase) -> TestCase:
        raw = await self.gateway.complete(
            module="judge",
            system_prompt=self._system,
            user_prompt=judge_user(
                session_trace_text=_trace_text(trace),
                test_case_text=_case_text(case),
            ),
        )
        case.verdict = _canon_verdict(raw.get("verdict"))
        case.judge_reason = str(raw.get("judge_reason", "")).strip()
        case.fault_type = _canon_fault(raw.get("fault_type"))
        return case


def _trace_text(trace: SessionTrace) -> str:
    lines = [f"会话 {trace.session_id}（{trace.domain} 域，{trace.turn_count} 轮）"]
    for e in trace.events:
        lines.append(f"  U: {e.user_utterance}")
        lines.append(f"  A: {e.agent_response}")
    return "\n".join(lines)


def _case_text(case: TestCase) -> str:
    return (
        f"用例 {case.case_id}（exp_mode={case.exp_mode}）\n"
        f"扰动输入: {case.disturb_utterance}\n"
        f"通过标准: {case.pass_criteria}\n"
        f"故障标准: {case.fail_criteria}"
    )
