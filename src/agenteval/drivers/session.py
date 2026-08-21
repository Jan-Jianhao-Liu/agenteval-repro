"""会话驱动器（L1）：多会话异步采集、会话重置、断点保存、死循环防护。"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Optional

from ..config import SessionConfig
from ..domain import DialogueEvent, SessionTrace
from ..storage import JsonlStore
from ..trace.logger import TraceLogger, get_logger, new_trace_id, set_trace_id
from .adapters import AgentAdapter

# 探索函数签名：给定会话轨迹/目标/轮次，返回下一条用户语句（None 结束会话）
ExploreFn = Callable[[SessionTrace, str, int], Awaitable[Optional[str]]]

_RECOVERY_UTTERANCES = [
    "你能处理退票业务吗？",
    "我想了解一下你们的改签政策",
    "请帮我查一下我的订单信息",
    "行李超重怎么收费？",
    "你好，还在吗？",
]


class SimpleExplorer:
    """默认探索规划器：按目标清单轮转提问 + 循环跟进，适合冒烟/离线验证。

    正式实验由 LLM 探索规划器（abstraction.prompter.explore_planner）替换，
    通过 SessionDriver(explore_fn=...) 注入，主流程零改动（开闭原则）。
    """

    def __init__(self, follow_ups: Optional[list[str]] = None):
        self.follow_ups = follow_ups or [
            "那退票的流程具体是怎样的？",
            "需要我提供什么信息？",
            "还有其他注意事项吗？",
        ]

    async def __call__(self, trace: SessionTrace, objective: str, turn_idx: int) -> str:
        # 首轮发目标问题，后续轮跟进
        if turn_idx == 0:
            return objective
        fu = self.follow_ups[(turn_idx - 1) % len(self.follow_ups)]
        return f"（针对刚才的话题）{fu}"


class SessionDriver:
    """驱动黑盒智能体完成多轮探索会话采集。"""

    def __init__(
        self,
        adapter: AgentAdapter,
        cfg: SessionConfig,
        store: Optional[JsonlStore] = None,
        logger: Optional[TraceLogger] = None,
        explore_fn: Optional[ExploreFn] = None,
    ):
        self.adapter = adapter
        self.cfg = cfg
        self.store = store
        self.logger = logger or get_logger()
        self.explore_fn = explore_fn or SimpleExplorer()

    async def run_rounds(self, objectives: list[str]) -> list[SessionTrace]:
        """并发执行多轮探索会话，返回全部会话轨迹（含断点落盘）。"""
        sem = asyncio.Semaphore(self.cfg.concurrency)
        tasks = [self._run_one(obj, sem) for obj in objectives]
        traces = await asyncio.gather(*tasks)
        return [t for t in traces if t is not None]

    async def _run_one(self, objective: str, sem: asyncio.Semaphore) -> Optional[SessionTrace]:
        async with sem:
            trace_id = new_trace_id()
            set_trace_id(trace_id)
            trace = SessionTrace(domain="airline", trace_id=trace_id)
            self.logger.log("session_driver", "INFO", input=objective, extra={"trace_id": trace_id, "event": "session_start"})
            try:
                await self.adapter.start_session()
                seen: dict[str, int] = {}  # 死循环检测：用户语句哈希 -> 次数
                for turn_idx in range(self.cfg.max_turns):
                    utterance = await self.explore_fn(trace, objective, turn_idx)
                    if utterance is None:
                        break

                    # 死循环防护：连续 3 次完全重复 → 强制随机扰动
                    h = hash(utterance)
                    seen[h] = seen.get(h, 0) + 1
                    if seen[h] >= 3:
                        utterance = random.choice(_RECOVERY_UTTERANCES)
                        seen.clear()
                        self.logger.log("session_driver", "WARN", input=utterance,
                                        extra={"event": "dead_loop_recovery"})

                    prev_context = trace.events[-1].agent_response if trace.events else ""
                    response = await self.adapter.send(utterance, prev_context=prev_context)
                    event = DialogueEvent(
                        turn_id=turn_idx,
                        user_utterance=utterance,
                        agent_response=response,
                    )
                    trace.add_event(event)
                    # 断点保存：每轮追加，进程中断不丢数据
                    if self.store:
                        self.store.append(f"sessions/{trace.session_id}.jsonl", trace)
                    self.logger.log("session_driver", "INFO",
                                    input=utterance, output=response[:500],
                                    extra={"session_id": trace.session_id, "turn": turn_idx})
                    if self._is_termination(response):
                        break
                trace.complete()
                if self.store:
                    self.store.append(f"sessions/{trace.session_id}.jsonl", trace)
                self.logger.log("session_driver", "INFO",
                                output=f"会话完成，共 {trace.turn_count} 轮",
                                extra={"session_id": trace.session_id})
                return trace
            except Exception as e:  # noqa: BLE001 单会话失败不影响整体实验
                trace.abort(reason=type(e).__name__)
                if self.store:
                    self.store.append(f"sessions/{trace.session_id}.jsonl", trace)
                self.logger.log("session_driver", "ERROR",
                                error=f"{type(e).__name__}: {e}",
                                extra={"session_id": trace.session_id, "event": "session_aborted"})
                if self.cfg.reset_on_error:
                    try:
                        await self.adapter.reset()
                    except Exception:  # noqa: BLE001
                        pass
                return None
            finally:
                set_trace_id("")

    @staticmethod
    def _is_termination(response: str) -> bool:
        """会话自然结束信号（Ollama 模拟客服可能主动结束）。"""
        endings = ("祝您旅途愉快", "感谢您的咨询", "再见", "还有其他需要吗")
        return any(e in response for e in endings)
