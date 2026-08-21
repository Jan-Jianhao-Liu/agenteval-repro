"""核心领域实体（轻量 DDD 五件套）。

全模块流转仅传递本文件定义的标准化实体，杜绝零散字典/原始文本跨层传递。
所有实体 Pydantic 强校验：残缺、格式异常数据直接拦截。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DialogueEvent(BaseModel):
    """单轮标准化：用户动作 + 智能体活动 + 语义合并键。"""

    model_config = ConfigDict(extra="forbid")

    turn_id: int = Field(ge=0, description="会话内轮次序号")
    user_utterance: str = Field(min_length=1, description="原始用户语句")
    agent_response: str = Field(min_length=1, description="原始智能体回复")
    user_action: Optional[str] = Field(default=None, description="标准化动作标签 动词_对象")
    agent_activity: Optional[str] = Field(default=None, description="标准化智能体活动标签 动词_对象")
    semantic_key: Optional[str] = Field(default=None, description="语义合并唯一标识，同义话术一致")

    @field_validator("user_action", "agent_activity", "semantic_key")
    @classmethod
    def _strip_tags(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        return v or None


class SessionTrace(BaseModel):
    """完整会话多轮对话记录（领域实体 1）。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(default_factory=lambda: _new_id("sess"))
    domain: str = Field(default="airline")
    events: list[DialogueEvent] = Field(default_factory=list)
    status: str = Field(default="active")  # active | completed | aborted
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    trace_id: Optional[str] = None

    def add_event(self, event: DialogueEvent) -> None:
        """追加单轮事件并刷新更新时间（断点保存友好）。"""
        self.events.append(event)
        self.updated_at = _now()

    def complete(self) -> None:
        self.status = "completed"
        self.updated_at = _now()

    def abort(self, reason: str = "") -> None:
        self.status = f"aborted:{reason}" if reason else "aborted"
        self.updated_at = _now()

    @property
    def turn_count(self) -> int:
        return len(self.events)


class WorkflowGraph(BaseModel):
    """DFG 对话工作流图（领域实体 3）：节点/边/频次，可双向转换 NetworkX。"""

    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field(default_factory=lambda: _new_id("dfg"))
    method: str = Field(default="full_method")  # prompt_only | graph_context | full_method
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)

    # ---- 序列化形态：nodes = [{id,label,count}] / edges = [{source,target,count}] ----

    def to_networkx(self) -> nx.DiGraph:
        """转换为 NetworkX 有向图（遍历/枚举/可视化）。"""
        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n["id"], label=n.get("label", n["id"]), count=n.get("count", 1))
        for e in self.edges:
            g.add_edge(e["source"], e["target"], count=e.get("count", 1))
        return g

    @classmethod
    def from_networkx(cls, g: nx.DiGraph, method: str = "full_method") -> "WorkflowGraph":
        nodes = [{"id": n, "label": d.get("label", n), "count": int(d.get("count", 1))} for n, d in g.nodes(data=True)]
        edges = [
            {"source": s, "target": t, "count": int(d.get("count", 1))}
            for s, t, d in g.edges(data=True)
        ]
        return cls(method=method, nodes=nodes, edges=edges)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_prompt_text(self) -> str:
        """Graph-context 消融模式用：图转纯文本。"""
        lines = [f"节点({len(self.nodes)}):"]
        for n in self.nodes:
            lines.append(f"  - {n['id']} (label={n.get('label', '')}, count={n.get('count', 1)})")
        lines.append(f"直接跟随边({len(self.edges)}):")
        for e in self.edges:
            lines.append(f"  - {e['source']} -> {e['target']} (count={e.get('count', 1)})")
        return "\n".join(lines)


class BoundaryTarget(BaseModel):
    """待测试边界位置、类型、上下文（领域实体 4）。"""

    model_config = ConfigDict(extra="forbid")

    boundary_id: str = Field(default_factory=lambda: _new_id("bnd"))
    guard_type: str = Field(description="确认门/身份校验/资格校验/非法输入校验")
    location: dict[str, str] = Field(default_factory=dict, description="node/edge 位置信息")
    context: str = Field(min_length=1, description="对应对话上下文摘要")
    potential: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM 打分 0~1")
    embedding: Optional[list[float]] = Field(default=None, description="聚类用 768 维向量")
    cluster_id: Optional[int] = Field(default=None, description="DBSCAN 聚类后簇号")


class TestCase(BaseModel):
    """标准化边界测试用例（领域实体 5）：扰动输入 + 通过/失败标准 + 判定结果。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(default_factory=lambda: _new_id("case"))
    boundary_id: str
    exp_mode: str = Field(default="full_method")
    disturb_utterance: str = Field(min_length=1, description="扰动用户语句")
    pass_criteria: str = Field(min_length=1)
    fail_criteria: str = Field(min_length=1)
    verdict: Optional[str] = Field(default=None)  # pass | fail | inconclusive
    judge_reason: Optional[str] = None
    fault_type: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    @field_validator("verdict")
    @classmethod
    def _valid_verdict(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower()
        if v not in {"pass", "fail", "inconclusive"}:
            raise ValueError(f"非法 verdict: {v}，仅允许 pass/fail/inconclusive")
        return v
