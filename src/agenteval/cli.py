"""CLI 主入口：python -m agenteval.cli --config config/dev.yaml --task smoke

task 支持：
  smoke   冒烟测试：会话采集 → 事件抽象 → DFG 构图，全链路验证
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import AppConfig
from .trace.logger import TraceLogger, configure


def _build_components(cfg: AppConfig):
    from .drivers import OllamaAgent, SessionDriver, SimpleExplorer, T3BenchAdapter
    from .llm import LLMCache, LLMGateway
    from .storage import JsonlStore

    trace_dir = Path(cfg.storage.trace_dir)
    logger = TraceLogger(trace_dir, save_input_output=cfg.trace.save_input_output)
    configure(logger)
    trace_dir.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(cfg.storage.data_dir)
    cache = LLMCache(cfg.cache.dir, enabled=cfg.cache.enabled)
    gateway = LLMGateway(cfg.llm, cache=cache, logger=logger, num_ctx=8192)

    if cfg.agent.type == "t3bench":
        adapter = T3BenchAdapter(
            domain=cfg.agent.domain,
            model=cfg.agent.model,
            temperature=cfg.agent.temperature,
            enable_dialog_cache=cfg.cache.enabled,
        )
    else:
        system_prompt = None
        if cfg.agent.system_prompt_file:
            sp = Path(cfg.agent.system_prompt_file)
            if not sp.is_absolute():
                sp = Path(__file__).resolve().parents[2] / cfg.agent.system_prompt_file
            if sp.exists():
                system_prompt = sp.read_text(encoding="utf-8")
        adapter = OllamaAgent(
            api_base=cfg.agent.api_base,
            model=cfg.agent.model,
            system_prompt=system_prompt,
            num_ctx=cfg.agent.num_ctx,
            temperature=cfg.agent.temperature,
            enable_dialog_cache=cfg.cache.enabled,
        )
    if cfg.session.explorer == "llm":
        from .drivers.llm_explorer import LLMExplorer

        explorer = LLMExplorer(gateway)
    else:
        explorer = SimpleExplorer()
    driver = SessionDriver(
        adapter=adapter,
        cfg=cfg.session,
        store=store,
        logger=logger,
        explore_fn=explorer,
    )
    return driver, gateway, store, logger


def _build_abstractor(cfg: AppConfig, gateway: "LLMGateway"):
    """构造事件抽象器（taxonomy + 语义映射 + 会话级缓存），供 smoke/dfg 复用。"""
    from .abstraction import AbstractCache, EventAbstractor
    from .abstraction.label_mapper import EmbedClient, LabelMapper, canonical_labels_from_smallset

    taxonomy: list[str] = []
    if cfg.taxonomy_file:
        sp = Path(cfg.taxonomy_file)
        if not sp.is_absolute():
            sp = Path(__file__).resolve().parents[2] / cfg.taxonomy_file
        if sp.exists():
            taxonomy = canonical_labels_from_smallset(sp)
    return EventAbstractor(
        gateway,
        cache=AbstractCache(Path(cfg.storage.data_dir) / "abstracted"),
        mapper=LabelMapper(
            canonical_labels=taxonomy,
            embed_client=EmbedClient(api_base=cfg.llm.api_base),
            threshold=0.82,
            cache_path=Path(cfg.storage.data_dir) / "embeddings" / "canonical_labels.json",
        ),
        taxonomy=taxonomy,
    )


async def run_smoke(cfg: AppConfig) -> int:
    """冒烟链路：探索会话 ×N → 事件抽象 → DFG 构图，输出摘要。"""
    driver, gateway, store, logger = _build_components(cfg)

    if cfg.session.objectives:
        objectives = cfg.session.objectives
    else:
        objectives = [
            "我想退掉下周三的机票",
            "帮我查一下我名下的订单",
            "航班延误了，改签要钱吗？",
            "我的行李超重了，怎么收费？",
            "帮我预订一张明天去北京的机票",
            "我的特价票可以退吗？",
            "帮我办理在线值机",
            "我想改签这周六的航班",
            "选座要额外收费吗？",
            "我的机票能升级商务舱吗？",
        ]
    objectives = objectives[: cfg.session.total_rounds]

    traces = await driver.run_rounds(objectives)

    from .abstraction.dfg_builder import DFGBuilder

    abstractor = _build_abstractor(cfg, gateway)
    abstracted = []
    for t in traces:
        if t is not None:
            abstracted.append(await abstractor.abstract_session(t))

    dfg = DFGBuilder().build(abstracted, method=cfg.exp_mode)

    print("=" * 60)
    print(f"[smoke] env={cfg.env_name} exp_mode={cfg.exp_mode}")
    print(f"[smoke] 完成会话: {len(traces)} 条（成功 {len(abstracted)} 条）")
    for t in abstracted:
        print(f"  - {t.session_id}: {t.turn_count} 轮, 状态={t.status}")
        for e in t.events:
            print(f"      [{e.turn_id}] {e.user_action} -> {e.agent_activity}  key={e.semantic_key}")
    print(f"[smoke] DFG 图: {dfg.node_count} 节点 / {dfg.edge_count} 边")
    print(f"[smoke] LLM 缓存: {gateway.cache.stats}")
    print(f"[smoke] Trace 目录: {cfg.storage.trace_dir}")
    print("=" * 60)
    return 0 if abstracted else 1


async def run_dfg(cfg: AppConfig) -> int:
    """复用已采集会话 → 抽象（缓存命中）→ DFG 构图 → SVG 可视化导出。

    注意：加载 sessions/*.jsonl 的「最后一行」（每轮断点都会写完整快照）。
    """
    import json

    from .abstraction.dfg_builder import DFGBuilder
    from .abstraction.dfg_render import render_svg
    from .domain import SessionTrace

    _, gateway, store, _ = _build_components(cfg)

    sess_dir = Path(cfg.storage.data_dir) / "sessions"
    traces: list[SessionTrace] = []
    if sess_dir.exists():
        for p in sorted(sess_dir.glob("*.jsonl")):
            lines = [l for l in p.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
            if lines:
                traces.append(SessionTrace.model_validate(json.loads(lines[-1])))
    if not traces:
        print(f"[dfg] 未找到已采集会话（{sess_dir}），请先运行 --task smoke")
        return 1

    abstractor = _build_abstractor(cfg, gateway)
    abstracted = [await abstractor.abstract_session(t) for t in traces]
    dfg = DFGBuilder().build(abstracted, method=cfg.exp_mode)
    out = render_svg(dfg, Path(cfg.storage.data_dir) / "dfg" / f"{dfg.graph_id}.svg")

    print("=" * 60)
    print(f"[dfg] 复用会话 {len(abstracted)} 条（共 {sum(t.turn_count for t in abstracted)} 轮）")
    print(f"[dfg] DFG 图: {dfg.node_count} 节点 / {dfg.edge_count} 边")
    print("[dfg] 节点明细:")
    for n in sorted(dfg.nodes, key=lambda x: -x.get("count", 1)):
        print(f"      {n['id']:<36} 频次 {n.get('count', 1)}")
    print(f"[dfg] SVG 已导出: {out}")
    print("=" * 60)
    return 0


def _load_traces(cfg: AppConfig) -> list["SessionTrace"]:
    """加载已采集探索会话（sessions/*.jsonl 最后一行快照）。"""
    import json

    from .domain import SessionTrace

    sess_dir = Path(cfg.storage.data_dir) / "sessions"
    traces: list[SessionTrace] = []
    if sess_dir.exists():
        for p in sorted(sess_dir.glob("*.jsonl")):
            lines = [l for l in p.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
            if lines:
                traces.append(SessionTrace.model_validate(json.loads(lines[-1])))
    return traces


async def run_experiment(cfg: AppConfig) -> int:
    """消融实验：三模式共享探索轨迹 → 测试生成 → 评测 → 指标对比。"""
    import json

    from .experiment import run_ablation
    from .experiment.metrics import compute_metrics

    driver, gateway, store, logger = _build_components(cfg)
    traces = _load_traces(cfg)
    if not traces:
        print("[experiment] 未找到已采集会话，请先运行 --task smoke 采集探索轨迹")
        return 1
    abstractor = _build_abstractor(cfg, gateway)

    from .abstraction.label_mapper import EmbedClient

    embed = EmbedClient(api_base=cfg.llm.api_base)
    # 审计器独立 LLM（如 GLM-5.2）：auditor_llm.model 非空时单独构建网关
    auditor_gateway = None
    if cfg.auditor_llm.model:
        from .llm import LLMCache, LLMGateway

        auditor_gateway = LLMGateway(
            cfg.auditor_llm,
            cache=LLMCache(cfg.cache.dir, enabled=cfg.cache.enabled),
            logger=logger,
        )
        print(f"[experiment] 审计器独立 LLM: {cfg.auditor_llm.provider}/{cfg.auditor_llm.model}")
    modes = getattr(cfg, "_modes", None)
    results = await run_ablation(
        cfg, driver.adapter, gateway, abstractor, store, logger, embed, traces,
        modes=modes, auditor_gateway=auditor_gateway,
    )

    # 验收判定（方案 5 Milestone 3，允许 ±5% 偏差）
    fm = results.get("full_method", {})
    if fm:
        ok_boundary = fm.get("n_independent_boundaries", 0) >= 20 * 0.95
        ok_dup = fm.get("dup_rate", 1.0) <= 0.30 * 1.05
        ok_coverage = fm.get("coverage_recall", 0) >= 0.92 * 0.95
        print("\n[experiment] Milestone 3 验收:")
        print(f"  full_method 独立边界 ≥20        : {'✅' if ok_boundary else '❌'} {fm.get('n_independent_boundaries', 0)}")
        print(f"  full_method 重复率 ≤0.30        : {'✅' if ok_dup else '❌'} {fm.get('dup_rate', 1):.3f}")
        print(f"  功能覆盖召回率 ≥0.92            : {'✅' if ok_coverage else '❌'} {fm.get('coverage_recall', 0):.3f}")
        if ok_boundary and ok_dup and ok_coverage:
            print("  总评: ✅ 全部达标（论文区间 23~38 独立边界 / Dup 0.26 可对照）")
        else:
            print("  总评: ❌ 未完全达标，请查看 results/ 明细")
        # 论文疑问点：Graph-context 边界数量 vs Prompt-only
        gc = results.get("graph_context", {}).get("n_independent_boundaries", 0)
        po = results.get("prompt_only", {}).get("n_independent_boundaries", 0)
        print(f"\n[experiment] 论文疑问验证: Graph-context({gc}) vs Prompt-only({po}) "
              f"-> {'Graph-context 更低，与论文一致' if gc < po else 'Graph-context 未低于 Prompt-only'}")
    return 0 if ok_boundary and ok_dup and ok_coverage else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="AGENTEVAL 复现工程 CLI")
    parser.add_argument("--config", default="config/dev.yaml", help="yaml 配置路径")
    parser.add_argument("--task", default="smoke", choices=["smoke", "dfg", "experiment"], help="执行任务")
    parser.add_argument("--exp-mode", default=None, choices=["prompt_only", "graph_context", "full_method"])
    parser.add_argument("--modes", default=None, help="消融模式子集，逗号分隔（如 full_method,graph_context）")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    if args.exp_mode:
        cfg.exp_mode = args.exp_mode

    if args.task == "smoke":
        return asyncio.run(run_smoke(cfg))
    if args.task == "dfg":
        return asyncio.run(run_dfg(cfg))
    if args.task == "experiment":
        if args.modes:
            cfg._modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        return asyncio.run(run_experiment(cfg))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
