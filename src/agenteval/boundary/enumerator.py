"""边界枚举器（L3）：DFG 节点/边 → 候选 BoundaryTarget。

纯图遍历规则（无 LLM）：
- 节点级：按标签关键词映射 guard_type（身份校验/资格校验/确认门/非法输入校验）；
- 边级：全部直接跟随边作为转移边界（前置步骤 → 后续步骤的校验跳转点）。
新增边界类型仅扩展 GUARD_KEYWORDS，不改动主流程（开闭原则）。
"""

from __future__ import annotations

from ..domain import BoundaryTarget, WorkflowGraph

# 节点标签关键词 → guard_type（顺序敏感：identity/eligibility 优先于 confirm 等泛词）
GUARD_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("identity",), "身份校验"),
    (("eligibility",), "资格校验"),
    (("confirm",), "确认门"),
    (("invalid", "retype"), "非法输入校验"),
]

# 全部合法 guard_type（打分器输出校准用）
GUARD_TYPES = ("确认门", "身份校验", "资格校验", "非法输入校验")


def guard_type_of(label: str) -> str | None:
    """由语义 key 推断节点所属边界类型；无校验语义返回 None（不做节点级边界）。"""
    for kws, gtype in GUARD_KEYWORDS:
        if any(k in label for k in kws):
            return gtype
    return None


class BoundaryEnumerator:
    """遍历 DFG 图生成候选边界：校验节点 + 全部转移边。"""

    def __init__(self, include_edges: bool = True):
        self.include_edges = include_edges

    def enumerate(self, dfg: WorkflowGraph) -> list[BoundaryTarget]:
        g = dfg.to_networkx()
        targets: list[BoundaryTarget] = []

        # 1) 节点级：含校验语义的节点
        for node_id, data in g.nodes(data=True):
            label = str(data.get("label", node_id))
            gtype = guard_type_of(label)
            if gtype is None:
                continue
            preds = [str(p) for p in g.predecessors(node_id)]
            succs = [str(s) for s in g.successors(node_id)]
            context = (
                f"节点 {label}（频次{data.get('count', 1)}）"
                f"；前置步骤: {preds or '无'}；后续步骤: {succs or '无'}"
            )
            targets.append(
                BoundaryTarget(
                    guard_type=gtype,
                    location={"kind": "node", "node_id": node_id},
                    context=context,
                )
            )

        # 2) 边级：全部直接跟随边（转移路径上的边界跳转点）
        if self.include_edges:
            for s, t, data in g.edges(data=True):
                targets.append(
                    BoundaryTarget(
                        guard_type="身份校验",  # 边级先统一按身份校验边界打分，由 LLM 校准
                        location={"kind": "edge", "source": str(s), "target": str(t)},
                        context=(
                            f"边 {s} -> {t}（频次{data.get('count', 1)}）："
                            f"用户从「{s}」直接推进到「{t}」的路径"
                        ),
                    )
                )
        return targets
