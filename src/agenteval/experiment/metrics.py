"""实验指标计算（方案 2.2 / 四、配套实验执行方案）。

指标定义：
- 独立边界数量：DBSCAN(eps, min_samples=1) 聚类后的簇数（boundary_embedding = guard_type+上下文）；
- 有效用例占比 valid_rate = verdict(inconclusive 之外) / 总数；
- 重复率 Dup = 1 - 独立边界数量 / 有效边界测试用例总数；
- 误报率 FAR = |judge=fail 且 audit=pass| / |judge=fail|（审计器校准 Judge 误报）；
- 准确率 = |judge==audit| / 总数；
- 功能覆盖召回率 = |测试会话覆盖节点 ∩ DFG 节点| / |DFG 节点|。
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..abstraction.label_mapper import EmbedClient
from ..domain import WorkflowGraph

EPS_DEFAULT = 0.7


async def cluster_boundaries(
    records: list[dict],
    embed: EmbedClient,
    eps: float = EPS_DEFAULT,
    min_samples: int = 1,
) -> None:
    """对每条记录计算 embedding 并 DBSCAN 聚类，写回 record['cluster_id']。

    方案 2.2：eps=0.7 为「相似度阈值」→ 转余弦距离 eps_dist = 1 - 0.7 = 0.3
    （DBSCAN metric=cosine 时 eps 是距离）。embedding 失败时 cluster_id=None。
    """
    if not records:
        return
    from sklearn.cluster import DBSCAN

    texts = [_cluster_text(r) for r in records]
    try:
        vectors = await embed.embed(texts)
    except Exception:  # noqa: BLE001 embedding 失败降级：cluster_id=None，实验不中断
        for r in records:
            r["cluster_id"] = None
        return
    eps_dist = max(0.01, 1.0 - eps)  # 相似度阈值 -> 余弦距离阈值
    model = DBSCAN(eps=eps_dist, min_samples=min_samples, metric="cosine")
    labels = model.fit_predict(vectors)
    for r, lab in zip(records, labels):
        r["cluster_id"] = int(lab) if lab >= 0 else None


def _cluster_text(r: dict) -> str:
    """聚类文本：guard_type + 目标图元素（高区分度特征）。

    弃用整段上下文（模板化措辞会把 bge-m3 向量压到一起，区分度差）。
    同时清洗数字/括号注释（本机 bge-m3 对特定长文本偶发 NaN）。
    """
    import re

    loc = r.get("location") or {}
    if loc.get("kind") == "node":
        target = f"节点 {loc.get('node_id', '')}"
    elif loc.get("kind") == "edge":
        target = f"边 {loc.get('source', '')} 到 {loc.get('target', '')}"
    else:
        target = str(r.get("context_text", ""))[:80]
    t = f"{r.get('guard_type', '')} {target}"
    t = re.sub(r"（[^）]*）", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:120]


def compute_metrics(
    records: list[dict],
    dfg: Optional[WorkflowGraph] = None,
    coverage_keys: Optional[set[str]] = None,
) -> dict:
    """聚合全部指标（纯函数，无副作用）。"""
    n_total = len(records)
    n_valid = sum(1 for r in records if r.get("verdict") != "inconclusive")
    clusters = {r.get("cluster_id") for r in records if r.get("cluster_id") is not None}
    n_independent = len(clusters)

    judge_fail = [r for r in records if r.get("judge_verdict") == "fail"]
    far = (
        sum(1 for r in judge_fail if r.get("verdict") == "pass") / len(judge_fail)
        if judge_fail
        else 0.0
    )
    consistent = sum(1 for r in records if r.get("verdict") == r.get("judge_verdict"))
    accuracy = consistent / n_total if n_total else 0.0

    coverage = 0.0
    if dfg is not None and coverage_keys is not None:
        node_ids = {n["id"] for n in dfg.nodes}
        covered = node_ids & coverage_keys
        coverage = len(covered) / len(node_ids) if node_ids else 0.0

    return {
        "exp_mode": records[0].get("exp_mode") if records else "",
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_rate": n_valid / n_total if n_total else 0.0,
        "n_independent_boundaries": n_independent,
        "dup_rate": 1 - (n_independent / n_valid) if n_valid else 0.0,
        "far": far,
        "accuracy": accuracy,
        "coverage_recall": coverage,
        "n_judge_fail": len(judge_fail),
    }


def format_metrics_table(metrics_list: Iterable[dict]) -> str:
    """三模式消融对比 Markdown 表格（自动生成，方案 3.3）。"""
    rows = list(metrics_list)
    header = (
        "| exp_mode | 独立边界 | 有效用例占比 | 重复率 | FAR | 准确率 | 覆盖召回率 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for m in rows:
        lines.append(
            f"| {m.get('exp_mode', '-')} | {m.get('n_independent_boundaries', 0)} "
            f"| {m.get('valid_rate', 0):.1%} | {m.get('dup_rate', 0):.3f} "
            f"| {m.get('far', 0):.3f} | {m.get('accuracy', 0):.1%} "
            f"| {m.get('coverage_recall', 0):.3f} |"
        )
    return "\n".join(lines)
