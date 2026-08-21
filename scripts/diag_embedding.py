"""诊断：LLM 原始标签 vs 规范标签的 bge-m3 余弦相似度 top3。

用于调 LabelMapper 阈值：确认是「阈值过松导致误映射」还是「正确标签相似度不足」。
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.abstraction.label_mapper import EmbedClient, canonical_labels_from_smallset  # noqa: E402

SMALLSET = Path(__file__).resolve().parents[1] / "data" / "smallset" / "airline_samples.json"
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "event_abstractor.jsonl"


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


async def main() -> None:
    labels = canonical_labels_from_smallset(SMALLSET)
    print(f"规范标签({len(labels)}): {labels}")

    # 从网关缓存读 LLM 原始输出标签
    raw_labels: list[str] = []
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out = rec.get("value") or {}
            for k in ("user_action", "agent_activity"):
                v = (out.get(k) or "").strip()
                if v and v not in raw_labels:
                    raw_labels.append(v)
    print(f"\nLLM 原始标签({len(raw_labels)}): {raw_labels}\n")

    embed = EmbedClient()
    vecs = await embed.embed(labels)
    raw_vecs = await embed.embed(raw_labels)

    print(f"{'原始标签':<34}{'top3 相似度(规范标签:sim)':<60}")
    print("-" * 100)
    for rl, rv in zip(raw_labels, raw_vecs):
        sims = sorted(
            ((c, _cos(rv, cv)) for c, cv in zip(labels, vecs)), key=lambda x: -x[1]
        )[:3]
        desc = "  ".join(f"{c}:{s:.3f}" for c, s in sims)
        print(f"{rl:<34}{desc}")
    await embed.close()


if __name__ == "__main__":
    asyncio.run(main())
