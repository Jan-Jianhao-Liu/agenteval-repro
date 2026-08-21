"""标准化 Prompt 模板库（L2）：五大模块统一强制纯 JSON 输出。

所有模板经 LLM 网关统一发送；JSON 结构固定，字段缺失即解析失败。
"""

from __future__ import annotations

import json
from typing import Optional

# ------------------------------------------------------------------ 通用约束

_JSON_ONLY = (
    "严格约束：只输出一个合法 JSON 对象，禁止 Markdown 代码块、禁止解释文字、"
    "禁止多余符号；字段缺失或格式错误将被判定为解析失败。"
)


def _system(role_desc: str, json_example: dict) -> str:
    return (
        f"{role_desc}\n\n{_JSON_ONLY}\n\n输出 JSON 结构示例（键名必须完全一致）：\n"
        f"{json.dumps(json_example, ensure_ascii=False)}"
    )


# ------------------------------------------------------------------ 探索规划器

EXPLORE_PLANNER_EXAMPLE = {
    "explore_objective": "本次会话探索目标简述",
    "user_utterance": "待发送用户对话语句",
    "risk_repeat": True,
}


def explore_planner_system() -> str:
    return _system(
        "你是探索规划器：基于已有对话轨迹挖掘尚未充分覆盖的业务流程，"
        "避免重复高频采集路径，输出下一条应发送给智能体的用户语句。",
        EXPLORE_PLANNER_EXAMPLE,
    )


def explore_planner_user(
    history: str, collected_activities: list[str], graph_summary: str = ""
) -> str:
    parts = [
        "【历史对话轨迹】\n" + (history or "（暂无，本次为首次探索）"),
        "【已采集活动清单】\n" + (", ".join(collected_activities) if collected_activities else "（暂无）"),
    ]
    if graph_summary:
        parts.append("【当前工作流图摘要】\n" + graph_summary)
    parts.append("请输出下一条探索语句。")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ 事件抽象器

EVENT_ABSTRACTOR_EXAMPLE = {
    "user_action": "标准化动作标签（动词_对象）",
    "agent_activity": "标准化智能体活动标签（动词_对象）",
    "semantic_key": "语义合并唯一标识字符串",
}


def event_abstractor_system(
    taxonomy: Optional[list[str]] = None,
    glossary: Optional[dict[str, str]] = None,
) -> str:
    base = (
        "你是事件抽象器：将单轮用户-智能体对话抽象为标准化动作标签。"
        "标签规则：[动词]_[业务对象]，例如 request_confirmation / list_reservation / deny_cancel。"
        "重要：semantic_key 必须等于 user_action 的标准化值（语义合并键 = 用户动作），"
        "同义不同话术必须输出完全一致的 user_action 与 semantic_key，用于图节点合并；"
        "仅参数（订单号/姓名等）不同的流程视为同一活动。"
    )
    if taxonomy:
        lines = []
        for label in taxonomy:
            gloss = f"（{glossary[label]}）" if glossary and label in glossary else ""
            lines.append(f"- {label}{gloss}")
        base += (
            "\n\n【可选动作标签表】必须从以下标签中选择语义最接近的一个，"
            "输出完全一致的字符串（不得自创、不得改写）：\n" + "\n".join(lines)
        )
    base += (
        "\n\n【示例】\n"
        '用户"我想退掉下周三的机票"，智能体"请提供订单号" → user_action=request_refund_ticket, '
        'agent_activity=request_order_info\n'
        '用户"确认退票"，智能体"已受理退票" → user_action=confirm_refund, agent_activity=execute_refund\n'
        "请严格参照示例与标签表输出。"
    )
    return _system(base, EVENT_ABSTRACTOR_EXAMPLE)


def event_abstractor_user(user_utterance: str, agent_response: str) -> str:
    return (
        "【用户语句】\n"
        f"{user_utterance}\n\n"
        "【智能体回复】\n"
        f"{agent_response}\n\n"
        "请抽象该轮对话的动作标签。"
    )


# ------------------------------------------------------------------ 边界打分器

BOUNDARY_SCORER_EXAMPLE = {
    "boundary_potential": 0.0,
    "guard_type": "确认门/身份校验/资格校验/非法输入校验",
    "perturb_suggest": "简短扰动思路描述",
}


def boundary_scorer_system() -> str:
    return _system(
        "你是边界打分器：评估工作流图中节点/边的边界测试潜力。"
        "guard_type 四选一：确认门、身份校验、资格校验、非法输入校验。"
        "boundary_potential 为 0~1 浮点数，越高代表越值得测试（如校验缺失、跳过确认等风险）。",
        BOUNDARY_SCORER_EXAMPLE,
    )


def boundary_scorer_user(
    node_or_edge_desc: str, context: str, tested_boundaries: list[str]
) -> str:
    return (
        "【待评估图元素】\n"
        f"{node_or_edge_desc}\n\n"
        "【对应对话上下文】\n"
        f"{context}\n\n"
        "【已测试边界】\n"
        + (", ".join(tested_boundaries) if tested_boundaries else "（无）")
        + "\n\n请输出该元素的边界潜力评估。"
    )


# ------------------------------------------------------------------ 边界测试生成器

TEST_GENERATOR_EXAMPLE = {
    "disturb_utterance": "用于测试的扰动用户语句",
    "pass_criteria": "测试通过判定标准",
    "fail_criteria": "边界故障判定标准",
}


def test_generator_system() -> str:
    return _system(
        "你是边界测试生成器：针对目标边界，基于正常前置对话路径构造扰动输入，"
        "并给出通过/失败判定标准。扰动需贴近真实用户话术，能触发边界校验逻辑。",
        TEST_GENERATOR_EXAMPLE,
    )


def test_generator_user(
    boundary_context: str, normal_path: str, perturb_type: str
) -> str:
    return (
        "【目标边界】\n"
        f"{boundary_context}\n\n"
        "【正常前置对话路径】\n"
        f"{normal_path}\n\n"
        f"【扰动类型】{perturb_type}\n\n"
        "请生成扰动用例。"
    )


# ------------------------------------------------------------------ 测试裁判 Judge

JUDGE_EXAMPLE = {
    "verdict": "pass/fail/inconclusive",
    "judge_reason": "完整判定依据",
    "fault_type": "无/跳过确认/身份缺失/资格未校验/非法参数放行",
}


def judge_system() -> str:
    return _system(
        "你是测试裁判：根据完整测试会话轨迹与测试用例预期标准，判定边界测试结果。"
        "verdict 三选一：pass（边界正确拦截/符合预期）、fail（发现边界故障）、inconclusive（无法判定）。"
        "fault_type 五选一：无、跳过确认、身份缺失、资格未校验、非法参数放行。",
        JUDGE_EXAMPLE,
    )


def judge_user(session_trace_text: str, test_case_text: str) -> str:
    return (
        "【完整测试会话轨迹】\n"
        f"{session_trace_text}\n\n"
        "【测试用例预期标准】\n"
        f"{test_case_text}\n\n"
        "请输出判定结果。"
    )


# ------------------------------------------------------------------ 模块注册表

PROMPT_FACTORIES = {
    "explore_planner": (explore_planner_system, explore_planner_user),
    "event_abstractor": (event_abstractor_system, event_abstractor_user),
    "boundary_scorer": (boundary_scorer_system, boundary_scorer_user),
    "test_generator": (test_generator_system, test_generator_user),
    "judge": (judge_system, judge_user),
}
