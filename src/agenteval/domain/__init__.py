"""领域实体层：全局统一 5 大核心领域实体（轻量 DDD，Pydantic 强校验）。"""

from .entities import (
    BoundaryTarget,
    DialogueEvent,
    SessionTrace,
    TestCase,
    WorkflowGraph,
)

__all__ = [
    "BoundaryTarget",
    "DialogueEvent",
    "SessionTrace",
    "TestCase",
    "WorkflowGraph",
]
