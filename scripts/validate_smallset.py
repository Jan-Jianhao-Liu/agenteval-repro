"""小样本验证脚本：对航空小样本跑事件抽象，输出验收指标报告。

指标（方案 2/方案 5 Milestone 2）：
- JSON 解析成功率：输出未落入兜底（unknown_action/unknown_activity）的轮次占比
- 标签匹配率：user_action / agent_activity 与 gold 严格一致的轮次占比
- semantic_key 合并正确率：输出 key 与 gold key 一致的轮次占比（同义话术应合并为同一节点）

用法：PYTHONPATH=src python scripts/validate_smallset.py --config config/dev.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.abstraction import AbstractCache, EventAbstractor  # noqa: E402
from agenteval.abstraction.label_mapper import (  # noqa: E402
    EmbedClient,
    LabelMapper,
    canonical_labels_from_smallset,
)
from agenteval.config import AppConfig  # noqa: E402
from agenteval.domain import DialogueEvent, SessionTrace  # noqa: E402
from agenteval.llm import LLMCache, LLMGateway  # noqa: E402
from agenteval.trace.logger import TraceLogger, configure  # noqa: E402

SMALLSET = Path(__file__).resolve().parents[1] / "data" / "smallset" / "airline_samples.json"

# 兜底标志（网关硬编码 fallback 输出）
_FALLBACK_FLAGS = {"unknown_action", "unknown_activity"}


def _build_trace(sample: dict) -> SessionTrace:
    return SessionTrace(
        session_id=f"gold_{sample['sample_id']}",
        domain="airline",
        status="completed",
        events=[
            DialogueEvent(turn_id=i, user_utterance=t["user"], agent_response=t["agent"])
            for i, t in enumerate(sample["turns"])
        ],
    )


def _is_fallback(val: str) -> bool:
    return val in _FALLBACK_FLAGS


async def run(cfg: AppConfig) -> int:
    samples = json.loads(SMALLSET.read_text(encoding="utf-8"))["samples"]
    logger = TraceLogger(Path(cfg.storage.trace_dir) / "validate", save_input_output=cfg.trace.save_input_output)
    configure(logger)
    gateway = LLMGateway(cfg.llm, cache=LLMCache(cfg.cache.dir, enabled=cfg.cache.enabled), logger=logger)
    embed = EmbedClient(api_base=cfg.llm.api_base)
    taxonomy = canonical_labels_from_smallset(SMALLSET)
    mapper = LabelMapper(
        canonical_labels=taxonomy,
        embed_client=embed,
        threshold=0.82,
        cache_path=Path(cfg.storage.data_dir) / "embeddings" / "canonical_labels.json",
    )
    abstractor = EventAbstractor(
        gateway,
        cache=AbstractCache(Path(cfg.storage.data_dir) / "abstracted"),
        mapper=mapper,
        taxonomy=taxonomy,
    )

    total = n_ok_json = n_ok_user = n_ok_agent = n_ok_key = 0
    print(f"{'样本':<6}{'轮':<4}{'user_action(预测/标准)':<38}{'agent_activity(预测/标准)':<38}{'key(预测/标准)':<34}判定")
    print("-" * 150)
    for sample in samples:
        trace = await abstractor.abstract_session(_build_trace(sample))
        for turn, event in zip(sample["turns"], trace.events):
            gold = turn["gold"]
            total += 1
            ok_json = not (_is_fallback(event.user_action) or _is_fallback(event.agent_activity))
            ok_user = event.user_action == gold["user_action"]
            ok_agent = event.agent_activity == gold["agent_activity"]
            ok_key = event.semantic_key == gold["semantic_key"]
            n_ok_json += ok_json
            n_ok_user += ok_user
            n_ok_agent += ok_agent
            n_ok_key += ok_key
            flags = f"{'JSON✓' if ok_json else 'JSON✗'}{' UA✓' if ok_user else ' UA✗'}{' AG✓' if ok_agent else ' AG✗'}{' KEY✓' if ok_key else ' KEY✗'}"
            print(
                f"{sample['sample_id']:<6}{event.turn_id:<4}"
                f"{event.user_action + ' / ' + gold['user_action']:<38}"
                f"{event.agent_activity + ' / ' + gold['agent_activity']:<38}"
                f"{str(event.semantic_key) + ' / ' + gold['semantic_key']:<34}{flags}"
            )
    print("-" * 150)
    print(f"总轮次: {total}")
    print(f"JSON 解析成功率: {n_ok_json / total:.1%}  ({n_ok_json}/{total})")
    print(f"user_action 匹配率: {n_ok_user / total:.1%}  ({n_ok_user}/{total})")
    print(f"agent_activity 匹配率: {n_ok_agent / total:.1%}  ({n_ok_agent}/{total})")
    print(f"semantic_key 合并正确率: {n_ok_key / total:.1%}  ({n_ok_key}/{total})  [目标 ≥90%]")
    ok = n_ok_json / total >= 0.95 and n_ok_key / total >= 0.90
    print(f"验收: {'✅ 通过' if ok else '❌ 未达标（检查标签粒度与同义合并）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小样本验证")
    parser.add_argument("--config", default="config/dev.yaml")
    args = parser.parse_args()
    cfg = AppConfig.load(args.config)
    raise SystemExit(asyncio.run(run(cfg)))
