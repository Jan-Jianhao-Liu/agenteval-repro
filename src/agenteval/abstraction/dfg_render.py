"""DFG 图 SVG 渲染器：纯 Python 分层布局，零重型依赖（不装 matplotlib/graphviz）。

布局：最长路径分层（源点层 0，节点层 = 最长入路径深度），层内纵向排列；
节点 = 椭圆（label + 出现频次），边 = 直线 + 箭头（直接跟随频次）。
浅色主题，适配学术文档/周报插图。
"""

from __future__ import annotations

import html
from pathlib import Path

from ..domain import WorkflowGraph

NODE_W = 170
NODE_H = 34
LAYER_GAP = 200
ROW_GAP = 78
MARGIN = 50


def _layout(g, start=(MARGIN, MARGIN)):
    """分层布局：返回 {node: (x, y)} 与图宽高。"""
    # 层号 = 最长入路径深度
    layer: dict = {}
    for n in g.nodes():
        preds = [layer[p] + 1 for p in g.predecessors(n) if p in layer]
        layer[n] = max(preds, default=0)
    # 逐层迭代修正（处理乱序；忽略自环，避免环导致层号无限递增）
    for _ in range(len(g)):
        changed = False
        for n in g.nodes():
            for p in g.predecessors(n):
                if p == n:
                    continue
                if layer[p] >= layer[n]:
                    layer[n] = layer[p] + 1
                    changed = True
        if not changed:
            break
    # 层内按节点原始顺序排布
    by_layer: dict[int, list] = {}
    for n in g.nodes():
        by_layer.setdefault(layer[n], []).append(n)
    pos: dict = {}
    for lyr, nodes in by_layer.items():
        for i, n in enumerate(nodes):
            pos[n] = (start[0] + lyr * LAYER_GAP, start[1] + i * ROW_GAP)
    max_layer = max(by_layer) if by_layer else 0
    width = start[0] + (max_layer + 1) * LAYER_GAP
    height = start[1] + max(len(v) for v in by_layer.values()) * ROW_GAP + MARGIN
    return pos, width, height


def render_svg(dfg: WorkflowGraph, out_path: str | Path) -> Path:
    """渲染 DFG 为 SVG 文件，返回输出路径。"""
    g = dfg.to_networkx()
    pos, width, height = _layout(g)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Microsoft YaHei, sans-serif">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" '
        'orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#8a6d3b"/></marker></defs>',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{MARGIN}" y="28" font-size="15" fill="#333333" font-weight="bold">'
        f'DFG 工作流图 · {dfg.node_count} 节点 / {dfg.edge_count} 边 · method={dfg.method}</text>',
    ]

    # 边（先画，避免盖住节点；自环画节点上方回环弧线）
    for s, t, d in g.edges(data=True):
        count = d.get("count", 1)
        if s == t:
            x, y = pos[s]
            parts.append(
                f'<path d="M {x - 34} {y - 20} C {x - 70} {y - 66}, {x + 70} {y - 66}, {x + 34} {y - 20}" '
                f'fill="none" stroke="#b0a088" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
            parts.append(
                f'<text x="{x}" y="{y - 52}" font-size="11" fill="#8a6d3b" text-anchor="middle">×{count}</text>'
            )
            continue
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#b0a088" '
            f'stroke-width="1.4" marker-end="url(#arrow)"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 6
        parts.append(
            f'<text x="{mx}" y="{my}" font-size="11" fill="#8a6d3b" text-anchor="middle">'
            f'×{count}</text>'
        )

    # 节点
    for n, d in g.nodes(data=True):
        x, y = pos[n]
        label = d.get("label", n)
        count = d.get("count", 1)
        rx = max(60, min(92, len(str(label)) * 6 + 18))
        parts.append(
            f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="18" fill="#eef2f7" '
            f'stroke="#5b7a9d" stroke-width="1.6"/>'
        )
        parts.append(
            f'<text x="{x}" y="{y - 2}" font-size="12" fill="#1f3b57" text-anchor="middle" '
            f'font-weight="bold">{html.escape(str(label))}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{y + 13}" font-size="10" fill="#7a8699" text-anchor="middle">'
            f'频次 {count}</text>'
        )

    parts.append("</svg>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
