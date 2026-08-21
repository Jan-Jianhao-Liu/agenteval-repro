"""消融实验流水线（方案 3）：三模式单一变量控制 + 全链路评测 + 指标聚合。

三模式（全局 yaml exp_mode 单一开关，仅测试生成分支不同）：
- prompt_only   : 不构建 DFG，仅历史对话摘要 → LLM 直接生成测试用例；
- graph_context : 构建 DFG 并转纯文本拼接进 Prompt（不结构化枚举）；
- full_method   : 结构化遍历图节点/边 → LLM 打分筛选 → 扰动生成（论文完整方案）。

实验一致性：三模式共享同一套探索会话轨迹、超参、Prompt 模板。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from ..abstraction.dfg_builder import DFGBuilder
from ..abstraction.event_abstractor import EventAbstractor
from ..abstraction.label_mapper import EmbedClient
from ..boundary import BoundaryEnumerator, BoundaryScorer, Perturber
from ..config import AppConfig
from ..domain import DialogueEvent, SessionTrace, TestCase
from ..evaluate import Auditor, LLMJudge
from ..llm import LLMGateway
from ..storage import JsonlStore
from ..trace.logger import TraceLogger, new_trace_id, set_trace_id
from .metrics import cluster_boundaries, compute_metrics, format_metrics_table

_MODES = ("prompt_only", "graph_context", "full_method")


def _infer_guard(text: str) -> str:
    """由扰动文本推断 guard_type（graph/prompt_only 模式无 LLM 打分时的归类）。"""
    if any(k in text for k in ("确认", "同意", "执行")):
        return "确认门"
    if any(k in text for k in ("身份", "姓名", "证件", "我没带", "忘了")):
        return "身份校验"
    if any(k in text for k in ("资格", "特价", "特惠", "免费", "不该")):
        return "资格校验"
    return "非法输入校验"


class ExperimentRunner:
    def __init__(
        self,
        cfg: AppConfig,
        adapter,
        gateway: LLMGateway,
        abstractor: EventAbstractor,
        store: JsonlStore,
        logger: TraceLogger,
        embed: EmbedClient,
        auditor_gateway=None,  # 审计器独立 LLM 网关（如 GLM-5.2）；None 复用主网关
    ):
        self.cfg = cfg
        self.adapter = adapter
        self.gateway = gateway
        self.abstractor = abstractor
        self.store = store
        self.logger = logger
        self.embed = embed
        self.judge = LLMJudge(gateway)
        self.auditor = Auditor(llm_judge=LLMJudge(auditor_gateway or gateway))
        self.enumerator = BoundaryEnumerator()
        self.scorer = BoundaryScorer(gateway, threshold=cfg.test.scorer_threshold)
        self.perturber = Perturber(gateway, max_cases=cfg.test.boundary_limit)
        self.sem = asyncio.Semaphore(cfg.session.concurrency)

    # ------------------------------------------------------------ 主入口

    async def run_mode(self, exp_mode: str, traces: list[SessionTrace]) -> dict:
        """跑单一消融模式，返回该模式指标 + 用例记录。"""
        # 覆盖写结果文件（幂等：重复运行不产生重复记录）
        result_file = Path(self.cfg.storage.data_dir) / "results" / f"{exp_mode}_cases.jsonl"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text("", encoding="utf-8")

        abstracted = [await self.abstractor.abstract_session(t) for t in traces]
        dfg = DFGBuilder().build(abstracted, method=exp_mode)

        cases = await self._gen_cases(exp_mode, dfg, abstracted)
        self.logger.log("experiment", "INFO",
                        input={"exp_mode": exp_mode, "cases": len(cases), "dfg_nodes": dfg.node_count})

        records: list[dict] = []
        coverage_keys: set[str] = set()
        for case in cases:
            rec = await self._run_one(case, dfg, coverage_keys)
            records.append(rec)

        await cluster_boundaries(records, self.embed, eps=self.cfg.dbscan.eps)
        metrics = compute_metrics(records, dfg=dfg, coverage_keys=coverage_keys)
        metrics["dfg_nodes"] = dfg.node_count
        metrics["dfg_edges"] = dfg.edge_count
        self._save(exp_mode, records, metrics, dfg)
        return metrics

    # ------------------------------------------------------------ 用例生成

    async def _gen_cases(
        self, exp_mode: str, dfg, abstracted: list[SessionTrace]
    ) -> list[TestCase]:
        limit = self.cfg.test.boundary_limit
        if exp_mode == "full_method":
            targets = self.enumerator.enumerate(dfg)
            scored = await self.scorer.score(targets)
            cases = []
            for t in scored[:limit]:
                case = await self.perturber.generate(t, exp_mode)
                if case is not None:
                    case_meta = _case_meta(case, t.guard_type, t.context, t.potential, t.location)
                    case.__dict__["_meta"] = case_meta  # 临时挂载（落盘时取用）
                    cases.append(case)
            return cases
        # prompt_only / graph_context：LLM 直接生成（输入 = 摘要或图文本）
        summary = _mode_summary(exp_mode, abstracted, dfg)
        cases = []
        for i in range(limit):
            raw = await self.gateway.complete(
                module="test_generator",
                system_prompt=self.perturber._system,
                user_prompt=(
                    f"【{exp_mode} 模式输入摘要】\n{summary}\n\n"
                    f"基于上述信息，生成第 {i + 1} 条边界测试扰动输入（优先覆盖尚未测试的边界）。\n"
                    "不要重复之前已生成的输入。"
                ),
            )
            utterance = str(raw.get("disturb_utterance", "")).strip()
            if not utterance:
                continue
            case = TestCase(
                boundary_id=f"bnd_llm_{exp_mode}_{i}",
                exp_mode=exp_mode,
                disturb_utterance=utterance,
                pass_criteria=str(raw.get("pass_criteria", "")).strip(),
                fail_criteria=str(raw.get("fail_criteria", "")).strip(),
            )
            case.__dict__["_meta"] = _case_meta(case, _infer_guard(utterance),
                                                summary[:200], None, {})
            cases.append(case)
            if len(cases) >= limit:
                break
        return cases

    # ------------------------------------------------------------ 单用例执行

    async def _run_one(self, case: TestCase, dfg, coverage_keys: set[str]) -> dict:
        async with self.sem:
            trace_id = new_trace_id()
            set_trace_id(trace_id)
            trace = await self._run_test_session(case)
            await self.judge.judge(trace, case)
            audit = await self.auditor.audit(trace, case, judge_verdict=case.verdict)
            # 测试会话抽象 → 覆盖节点（功能覆盖召回率）
            await self.abstractor.abstract_session(trace)
            for e in trace.events:
                if e.semantic_key:
                    coverage_keys.add(e.semantic_key)

            meta = getattr(case, "_meta", {})
            # 覆盖统计：full_method 用用例瞄准的图元素；其他模式用测试会话语义 key
            loc = meta.get("location") or {}
            if loc.get("kind") == "node":
                coverage_keys.add(loc["node_id"])
            elif loc.get("kind") == "edge":
                coverage_keys.add(loc["source"])
                coverage_keys.add(loc["target"])
            rec = {
                "trace_id": trace_id,
                "case_id": case.case_id,
                "exp_mode": case.exp_mode,
                "boundary_id": case.boundary_id,
                "guard_type": meta.get("guard_type", "非法输入校验"),
                "context_text": meta.get("context", ""),
                "location": meta.get("location", {}),
                "potential": meta.get("potential"),
                "disturb_utterance": case.disturb_utterance,
                "agent_response": trace.events[-1].agent_response[:500] if trace.events else "",
                "judge_verdict": case.verdict,
                "verdict": audit["verdict"],
                "fault_type": audit["fault_type"],
                "audit_source": audit["source"],
                "reason": audit["reason"],
                "cluster_id": None,
            }
            self.store.append(f"results/{case.exp_mode}_cases.jsonl", rec)
            self.logger.log("experiment", "INFO",
                            input={"case": case.case_id, "utterance": case.disturb_utterance},
                            output={"judge": rec["judge_verdict"], "audit": rec["verdict"], "fault": rec["fault_type"]},
                            extra={"exp_mode": case.exp_mode})
            return rec

    async def _run_test_session(self, case: TestCase) -> SessionTrace:
        """单轮扰动测试：发送扰动输入，收集 agent 回复后结束。"""
        await self.adapter.start_session()
        try:
            response = await self.adapter.send(case.disturb_utterance)
        finally:
            await self.adapter.reset()
        trace = SessionTrace(
            domain="airline", status="completed",
            events=[DialogueEvent(turn_id=0, user_utterance=case.disturb_utterance, agent_response=response)],
        )
        return trace

    # ------------------------------------------------------------ 落盘

    def _save(self, exp_mode: str, records: list[dict], metrics: dict, dfg) -> None:
        out = Path(self.cfg.storage.data_dir) / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{exp_mode}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.logger.log("experiment", "INFO",
                        input={"exp_mode": exp_mode}, output=metrics)


def _case_meta(case: TestCase, guard_type: str, context: str, potential, location=None) -> dict:
    return {
        "guard_type": guard_type,
        "context": context,
        "potential": potential,
        "location": location or {},
    }


def _mode_summary(exp_mode: str, abstracted: list[SessionTrace], dfg) -> str:
    """prompt_only 用历史对话摘要；graph_context 用图文本（方案 3.1）。"""
    if exp_mode == "graph_context":
        return dfg.to_prompt_text()
    lines = [f"探索会话 {len(abstracted)} 条："]
    for t in abstracted:
        for e in t.events[:6]:
            lines.append(f"  U: {e.user_utterance[:120]}")
            lines.append(f"  A: {e.agent_response[:120]}")
    return "\n".join(lines)[:3000]


async def run_ablation(
    cfg: AppConfig,
    adapter,
    gateway: LLMGateway,
    abstractor: EventAbstractor,
    store: JsonlStore,
    logger: TraceLogger,
    embed: EmbedClient,
    traces: list[SessionTrace],
    modes: Optional[list[str]] = None,
    auditor_gateway=None,
) -> dict:
    """顺序跑三组消融（共享探索轨迹），返回 {mode: metrics} 并打印对比表。"""
    runner = ExperimentRunner(cfg, adapter, gateway, abstractor, store, logger, embed,
                              auditor_gateway=auditor_gateway)
    results: dict[str, dict] = {}
    for mode in (modes or _MODES):
        print(f"\n[ablation] 运行模式: {mode} ...")
        metrics = await runner.run_mode(mode, traces)
        results[mode] = metrics
        m = metrics
        print(f"  -> 用例 {m['n_total']}（有效 {m['n_valid']}）| 独立边界 {m['n_independent_boundaries']} "
              f"| 重复率 {m['dup_rate']:.3f} | FAR {m['far']:.3f} | 覆盖召回 {m['coverage_recall']:.3f}")
    table = format_metrics_table(list(results.values()))
    out = Path(cfg.storage.data_dir) / "results" / "ablation_metrics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# 消融实验对比（{cfg.env_name} 环境）\n\n{table}\n", encoding="utf-8")
    print("\n[ablation] 消融对比表:")
    print(table)
    print(f"[ablation] 已保存: {out}")
    return results
