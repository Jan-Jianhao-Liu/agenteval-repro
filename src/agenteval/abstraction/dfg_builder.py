"""DFG 对话工作流图构建器（L2）：多条会话轨迹 → 直接跟随有向图。

节点 = 语义合并后的活动标签（semantic_key），边 = 轮次间的直接跟随关系，
均累计频次。图实体序列化为 WorkflowGraph（可转 NetworkX 遍历/可视化）。
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from ..domain import SessionTrace, WorkflowGraph


class DFGBuilder:
    """聚合多条会话轨迹生成 DFG。"""

    def build(self, traces: list[SessionTrace], method: str = "full_method") -> WorkflowGraph:
        g = nx.DiGraph()
        node_count: Counter = Counter()
        edge_count: Counter = Counter()

        for trace in traces:
            keys = [e.semantic_key for e in trace.events if e.semantic_key]
            if not keys:
                continue
            for k in keys:
                node_count[k] += 1
            for a, b in zip(keys, keys[1:]):
                edge_count[(a, b)] += 1

        for k, c in node_count.items():
            g.add_node(k, label=k, count=c)
        for (a, b), c in edge_count.items():
            g.add_edge(a, b, count=c)

        return WorkflowGraph.from_networkx(g, method=method)
