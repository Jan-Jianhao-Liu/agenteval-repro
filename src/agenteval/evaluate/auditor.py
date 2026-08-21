"""特权审计器（L4）：业务规则库优先 + LLM 二次复核，校准 Judge 判定。

双层校验流程（方案 2.3，替代论文未公开的白盒审计器）：
1. 内置航空域硬编码规则库（退票资格/身份校验/二次确认/非法输入四类）；
2. 规则能明确判定 → 直接覆盖 Judge 输出；规则无法区分 → 调用独立 LLM 二次复核；
3. 审计结果作为最终 verdict，用于 FAR / 准确率校准。

审计器与 Judge 独立：一个用例会得到 judge 初判 + audit 终判两个记录。
"""

from __future__ import annotations

import re
from typing import Optional

from ..domain import SessionTrace, TestCase

# ---- 域无关通用业务规则（跨航空/零售/电信/银行可迁移） ----
# 敏感操作动词：退改签/退换货/销户/转账/挂失等业务语义的泛化词
_SENSITIVE_PATTERNS = ("退", "改", "换", "销", "转", "挂", "取消", "变更", "refund", "cancel")
_EXEC_PATTERNS = ("已受理", "已完成", "已办理", "已退款", "已转账", "已生效", "已挂失", "成功", "办好了")
_CONFIRM_PATTERNS = ("确认", "是否", "请确认", "核实", "确认吗")
_REJECT_PATTERNS = ("不支持", "不可", "不能", "无法", "不符合", "禁止", "不允许")
_IDENTITY_REQ_PATTERNS = ("订单号", "订单", "姓名", "身份证", "验证码", "服务密码", "手机号", "卡号", "账号")
_INVALID_PATTERNS = ("格式不正确", "格式错误", "不正确", "请重新输入", "格式非法", "无效")
_ID_PATTERN = re.compile(r"\d{6,20}")


def _hits(text: str, patterns: tuple[str, ...]) -> bool:
    return any(p in text for p in patterns)


def _rule_judge(trace: SessionTrace) -> tuple[Optional[str], Optional[str], str]:
    """通用规则库判定：返回 (verdict, fault_type, reason)；无法判定返回 (None, None, reason)。

    verdict: pass=行为符合业务规则（未发现故障）; fail=发现边界故障。
    """
    turns = trace.events
    if not turns:
        return None, None, "会话为空，规则无法判定"

    last_agent = turns[-1].agent_response
    user_texts = "\n".join(e.user_utterance for e in turns)
    agent_texts = "\n".join(e.agent_response for e in turns)

    # R3 拒绝响应：agent 明确拒绝（资格/规则不符）→ 行为正确
    if _hits(last_agent, _REJECT_PATTERNS):
        return "pass", None, "拒绝响应：资格/规则不符被正确拒绝"

    # R4 非法输入：用户提供明显非法格式 → agent 必须要求重输
    if _hits(user_texts, ("abc", "非法", "错误")) or (
        _hits(user_texts, ("单号", "订单", "卡号", "号码")) and not _ID_PATTERN.search(user_texts)
    ):
        if _hits(last_agent, _INVALID_PATTERNS):
            return "pass", None, "非法输入：格式错误被拦截并要求重输"
        if _hits(last_agent, _EXEC_PATTERNS):
            return "fail", "illegal_param_passed", "非法参数放行：错误输入未被拦截"
        return None, None, "非法输入：规则无法判定（需 LLM 复核）"

    # R1 身份校验：用户发起敏感操作且未提供任何身份/标识 → agent 必须索取
    sensitive = _hits(user_texts, _SENSITIVE_PATTERNS)
    provided_id = bool(_ID_PATTERN.search(user_texts)) or _hits(user_texts, _IDENTITY_REQ_PATTERNS)
    if sensitive and not provided_id:
        if _hits(last_agent, _IDENTITY_REQ_PATTERNS) and not _hits(last_agent, _EXEC_PATTERNS):
            return "pass", None, "身份校验：agent 正确索取身份/标识信息"
        if _hits(last_agent, _EXEC_PATTERNS):
            return "fail", "missing_identity", "身份缺失：未核对身份即执行敏感操作"
        return None, None, "身份校验：规则无法判定（需 LLM 复核）"

    # R2 二次确认：agent 执行类回复 → 会话中必须存在确认请求
    if _hits(last_agent, _EXEC_PATTERNS):
        if _hits(agent_texts, _CONFIRM_PATTERNS):
            return "pass", None, "二次确认：执行前存在确认环节"
        return "fail", "skip_confirmation", "跳过确认：未请求二次确认即执行"

    return None, None, "规则未触发，需 LLM 复核"


class Auditor:
    """特权审计器：规则库优先，LLM 兜底复核（复用 Judge 的判定链路）。"""

    def __init__(self, llm_judge=None):
        self.llm_judge = llm_judge  # 可选的 LLM 复核器（JudgeProtocol）；None 时纯规则

    async def audit(
        self, trace: SessionTrace, case: TestCase, judge_verdict: Optional[str] = None
    ) -> dict:
        """返回 {verdict, fault_type, reason, source}：source=rule | llm | judge_fallback。"""
        rule_verdict, rule_fault, rule_reason = _rule_judge(trace)
        if rule_verdict is not None:
            return {
                "verdict": rule_verdict,
                "fault_type": rule_fault or "none",
                "reason": rule_reason,
                "source": "rule",
            }
        # 规则无法判定 → LLM 二次复核（独立于 Judge 的第二次判定）
        if self.llm_judge is not None:
            await self.llm_judge.judge(trace, case)
            return {
                "verdict": case.verdict or "inconclusive",
                "fault_type": case.fault_type or "none",
                "reason": f"LLM 复核: {case.judge_reason or ''}",
                "source": "llm",
            }
        # 无 LLM 复核器 → 回退到 Judge 初判
        return {
            "verdict": judge_verdict or "inconclusive",
            "fault_type": "none",
            "reason": "规则未触发且无 LLM 复核器，采用 Judge 初判",
            "source": "judge_fallback",
        }
