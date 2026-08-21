"""语义标签规范化器：LLM 原始标签 → 规范标签空间（语义合并核心）。

两级映射（方案 2.1「同义不同话术输出完全一致 semantic_key」落地）：
1. 字符串快速路径：token 规范化后集合相等（覆盖 request_ticket_refund vs request_refund_ticket）；
2. Embedding 语义路径：bge-m3 向量余弦相似度，超过阈值映射到规范标签
   （覆盖 cancel vs refund、request_confirmation vs confirm_refund 等同义词）。

映射失败的标签保留原样（进 DFG 前仍是有效节点，仅不参与规范化合并）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import httpx


class EmbedClient:
    """Ollama /api/embed 封装（bge-m3，1024 维）。

    分批请求 + 错误重试（模型切换加载竞态常见 500/429）。
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:11434",
        model: str = "bge-m3",
        batch_size: int = 8,
        retries: int = 2,
        backoff: float = 3.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.retries = retries
        self.backoff = backoff
        self._client: Optional[httpx.AsyncClient] = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=180.0)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(await self._embed_batch(batch))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.post(
                    f"{self.api_base}/api/embed",
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                vecs = list(resp.json()["embeddings"])
                # bge-m3 偶发 NaN 向量（Ollama 500 的另一种形态）：检出即重试
                if any(math.isnan(x) for v in vecs for x in v):
                    raise ValueError("embedding 含 NaN，重试")
                return vecs
            except Exception as e:  # noqa: BLE001 500/429/NaN/超时等统一重试
                last_err = e
                await asyncio.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(f"embedding 请求失败({self.retries + 1} 次): {last_err}")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _tokens(label: str) -> frozenset[str]:
    """标签 token 规范化：小写、去分隔符。"""
    return frozenset(label.lower().replace("-", "_").replace(" ", "_").split("_"))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LabelMapper:
    """规范化 LLM 标签到规范标签空间。"""

    def __init__(
        self,
        canonical_labels: list[str],
        embed_client: EmbedClient,
        threshold: float = 0.78,
        cache_path: str | Path = "data/embeddings/canonical_labels.json",
    ):
        self.canonical = canonical_labels
        self.embed = embed_client
        self.threshold = threshold
        self.cache_path = Path(cache_path)
        self._vectors: Optional[list[list[float]]] = None

    # ---- 缓存：规范标签向量首次计算后落盘 ----

    async def _canonical_vectors(self) -> list[list[float]]:
        if self._vectors is not None:
            return self._vectors
        key = hashlib.sha256(json.dumps(self.canonical, ensure_ascii=False).encode()).hexdigest()[:16]
        if self.cache_path.exists():
            try:
                rec = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if rec.get("key") == key and len(rec["vectors"]) == len(self.canonical):
                    self._vectors = rec["vectors"]
                    return self._vectors
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        vecs = await self.embed.embed(self.canonical)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps({"key": key, "labels": self.canonical, "vectors": vecs}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._vectors = vecs
        return vecs

    # ---- 主入口 ----

    async def map_label(self, raw: str) -> str:
        """映射单个标签；失败保留原样。"""
        raw = raw.strip().lower()
        if not raw or raw in {"unknown_action", "unknown_activity"}:
            return raw
        # 1) 字符串快速路径：token 集合相等（词序变体）
        toks = _tokens(raw)
        for c in self.canonical:
            if _tokens(c) == toks:
                return c
        # 2) Embedding 语义路径
        vec = (await self.embed.embed([raw]))[0]
        cvecs = await self._canonical_vectors()
        best, best_sim = None, 0.0
        for c, cv in zip(self.canonical, cvecs):
            s = _cosine(vec, cv)
            if s > best_sim:
                best, best_sim = c, s
        if best is not None and best_sim >= self.threshold:
            return best
        return raw


def canonical_labels_from_smallset(smallset_path: str | Path) -> list[str]:
    """从航空小样本 gold 提取规范标签集（去重、保序）。"""
    rec = json.loads(Path(smallset_path).read_text(encoding="utf-8"))
    labels: list[str] = []
    seen: set[str] = set()
    for s in rec["samples"]:
        for t in s["turns"]:
            for v in t["gold"].values():
                v = v.strip()
                if v and v not in seen:
                    seen.add(v)
                    labels.append(v)
    return labels


# 标签含义注释（注入 Prompt 帮助小模型区分角色/粒度）
LABEL_GLOSSARY: dict[str, str] = {
    "request_refund_ticket": "用户请求退票/取消航班",
    "request_order_info": "智能体索取订单号",
    "provide_order_info": "用户提供订单信息（订单号/航班号）",
    "verify_identity": "智能体核对用户身份",
    "provide_identity": "用户提供身份信息（姓名/证件）",
    "check_refund_eligibility": "智能体校验退票资格并告知结果",
    "confirm_refund": "用户确认退票",
    "request_refund_confirmation": "智能体请求二次确认退票",
    "execute_refund": "智能体执行退票并告知结果",
    "withdraw_refund_request": "用户放弃退票请求",
    "provide_alternative": "智能体提供替代方案（改签/保留）",
    "provide_invalid_order": "用户提供非法/格式错误的订单号",
    "request_retype_order": "智能体要求重新输入订单号",
    "request_order_by_identity": "用户请求按身份信息查询订单",
    "request_identity_info": "智能体索取身份信息",
    "confirm_order": "用户确认订单信息",
}
