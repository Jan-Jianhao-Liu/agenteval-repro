"""用小样本验证集（gold 对话）构建 DFG 并渲染 SVG：周报/论文素材图。

用法：PYTHONPATH=src python scripts/render_smallset_dfg.py --config config/dev.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.abstraction import AbstractCache, EventAbstractor  # noqa: E402
from agenteval.abstraction.dfg_builder import DFGBuilder  # noqa: E402
from agenteval.abstraction.dfg_render import render_svg  # noqa: E402
from agenteval.abstraction.label_mapper import (  # noqa: E402
    EmbedClient,
    LabelMapper,
    canonical_labels_from_smallset,
)
from agenteval.config import AppConfig  # noqa: E402
from agenteval.domain import DialogueEvent, SessionTrace  # noqa: E402
from agenteval.llm import LLMCache, LLMGateway  # noqa: E402

SMALLSET = Path(__file__).resolve().parents[1] / "data" / "smallset" / "airline_samples.json"


async def run(cfg: AppConfig) -> int:
    samples = json.loads(SMALLSET.read_text(encoding="utf-8"))["samples"]
    traces = [
        SessionTrace(
            session_id=f"gold_{s['sample_id']}",
            domain="airline",
            status="completed",
            events=[
                DialogueEvent(turn_id=i, user_utterance=t["user"], agent_response=t["agent"])
                for i, t in enumerate(s["turns"])
            ],
        )
        for s in samples
    ]
    gateway = LLMGateway(cfg.llm, cache=LLMCache(cfg.cache.dir, enabled=cfg.cache.enabled))
    taxonomy = canonical_labels_from_smallset(SMALLSET)
    abstractor = EventAbstractor(
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
    abstracted = [await abstractor.abstract_session(t) for t in traces]
    dfg = DFGBuilder().build(abstracted, method="full_method")
    out = render_svg(dfg, Path(cfg.storage.data_dir) / "dfg" / f"smallset_{dfg.graph_id}.svg")
    print(f"小样本 DFG: {dfg.node_count} 节点 / {dfg.edge_count} 边 -> {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dev.yaml")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(AppConfig.load(args.config))))
