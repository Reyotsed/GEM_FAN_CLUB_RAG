"""Photo repository — fetches the curated GEM photo pool from the Java
backend, builds an in-memory **hybrid index** (BM25 + caption_zh embedding)
over (note + tags + caption + dims), and answers tag/keyword queries.

Lifecycle::

    repo = PhotoRepository(base_url=..., sync_key=..., ttl=600, embeddings=...)
    await repo.start()      # initial fetch + schedule refresh
    hits = repo.search("演唱会 红裙", top_k=3)
    await repo.stop()

Notes
-----
- Refresh is best-effort: on failure, the previous index stays usable.
- Hybrid retrieval: BM25 (lexical) + caption embedding (semantic), fused via
  Reciprocal Rank Fusion (RRF). Falls back to pure BM25 when ``embeddings``
  is not injected — backward compatible.
- Search is fully synchronous (uses pre-built indexes), so it can be invoked
  from a sync Skill handler without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx
from rank_bm25 import BM25Okapi

from app.core.text_utils import chinese_preprocess_func

logger = logging.getLogger(__name__)


_HASHTAG_RE = re.compile(r"#([\u4e00-\u9fffA-Za-z0-9]+)")


@dataclass(frozen=True)
class _Photo:
    url: str
    note: str
    caption_zh: str
    tags: tuple[str, ...]
    dims: Dict[str, Any]
    score: float  # VLM score 0..10


def _extract_tags(note: str) -> List[str]:
    """Pull `#tag` tokens out of the note column for backward compatibility."""
    if not note:
        return []
    return _HASHTAG_RE.findall(note)


def _loads_json(value: Any, fallback: Any) -> Any:
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _flatten_dim_terms(value: Any) -> List[str]:
    terms: List[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            terms.append(str(key))
            terms.extend(_flatten_dim_terms(nested))
    elif isinstance(value, list):
        for nested in value:
            terms.extend(_flatten_dim_terms(nested))
    elif value is not None:
        text = str(value).strip()
        if text:
            terms.append(text)
    return terms


def _tokenize_for_index(p: _Photo) -> List[str]:
    """Build the indexed token bag for a single photo.

    Strategy:
      • Tags get **3x weight** (repeated 3 times in the bag) — they are
        curated, high-precision signals.
      • Plain note text is jieba-tokenised once.
    """
    bag: List[str] = []
    for t in p.tags:
        bag.extend([t, t, t])
    for t in _flatten_dim_terms(p.dims):
        bag.extend([t, t])
    bag.extend(chinese_preprocess_func(p.caption_zh or ""))
    bag.extend(chinese_preprocess_func(p.note or ""))
    return bag


class PhotoRepository:
    """Fetch + cache + hybrid-index the curated GEM photo pool.

    Hybrid retrieval = BM25 (lexical, from tags/note/caption) + caption
    embedding (semantic). Fused via RRF.
    """

    def __init__(
        self,
        *,
        base_url: str,
        sync_key: Optional[str] = None,
        ttl_seconds: int = 600,
        timeout_seconds: float = 5.0,
        min_score: float = 0.0,
        embeddings: Optional[Any] = None,
        rrf_k: int = 60,
        vlm_score_weight: float = 0.05,
    ) -> None:
        if not base_url:
            raise ValueError("PhotoRepository requires a non-empty base_url")
        # Strip trailing slash for clean URL joining
        self._base_url = base_url.rstrip("/")
        self._sync_key = sync_key or ""
        self._ttl = max(60, int(ttl_seconds))
        self._timeout = float(timeout_seconds)
        self._min_score = float(min_score)
        # Embedding object: must implement ``embed_documents(List[str]) -> List[List[float]]``
        # and ``embed_query(str) -> List[float]``. ``ZhipuAIEmbeddings`` satisfies both.
        # ``None`` means semantic recall is disabled (pure BM25).
        self._embeddings = embeddings
        self._rrf_k = max(1, int(rrf_k))
        self._vlm_score_weight = float(vlm_score_weight)

        # Index state — protected by a simple lock; readers grab a snapshot.
        self._lock = threading.RLock()
        self._photos: List[_Photo] = []
        self._bm25: Optional[BM25Okapi] = None
        # numpy-friendly L2-normalized embedding matrix, shape (N, D).
        # ``None`` when embeddings is disabled or load failed.
        self._embed_matrix: Optional[Any] = None
        self._last_loaded_at: float = 0.0

        # Background refresh task
        self._refresh_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """One-shot load on startup. No periodic refresh.

        Why no auto-refresh:
        - Each refresh re-embeds all photo captions (50 × embedding API calls),
          which is non-trivial cost and latency.
        - Photo pool only changes when DATA pipeline runs (manual trigger),
          so a stale TTL window is wasteful.
        - Trade-off: fresh photos require restarting the RAG service after
          DATA pipeline syncs. Acceptable since pipeline runs are infrequent.

        To pick up new photos: restart the RAG service.
        """
        await self._load_safe()
        # Refresh task intentionally not scheduled.
        self._stop_event = None
        self._refresh_task = None

    async def stop(self) -> None:
        # Refresh loop is no longer scheduled, but keep the cleanup defensive
        # in case future code re-enables it.
        if self._stop_event is not None:
            self._stop_event.set()
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._refresh_task = None
        self._stop_event = None

    async def _load_safe(self) -> None:
        try:
            await self._load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("photo_repo.load_failed error=%s", exc)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def _load(self) -> None:
        url = f"{self._base_url}/gem-photo/list"
        headers = {"X-Sync-Key": self._sync_key} if self._sync_key else None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        if not isinstance(payload, dict) or payload.get("code") != 200:
            raise RuntimeError(f"unexpected payload from {url}: {str(payload)[:200]}")

        items = (payload.get("data") or {}).get("items") or []
        photos: List[_Photo] = []
        for it in items:
            cdn = it.get("url")
            if not cdn:
                continue
            note = (it.get("note") or "").strip()
            caption_zh = (it.get("captionZh") or it.get("caption_zh") or "").strip()
            tags_raw = _loads_json(it.get("tagsJson") or it.get("tags_json"), [])
            dims = _loads_json(it.get("dimsJson") or it.get("dims_json"), {})
            tags = [str(t).strip() for t in tags_raw if str(t).strip()] if isinstance(tags_raw, list) else []
            if not tags:
                tags = _extract_tags(note)
            score_raw = it.get("score")
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            if score < self._min_score:
                continue

            # Quality gate: skip photos that LLM can't understand
            has_caption = bool(caption_zh)
            has_tags = len(tags) > 0
            if not has_caption and not has_tags:
                continue

            photos.append(_Photo(url=cdn, note=note, caption_zh=caption_zh, tags=tuple(tags), dims=dims if isinstance(dims, dict) else {}, score=score))

        if not photos:
            logger.warning("photo_repo.loaded empty pool")

        # Build BM25 over the new snapshot
        corpus_tokens = [_tokenize_for_index(p) for p in photos] if photos else []
        bm25 = BM25Okapi(corpus_tokens) if any(corpus_tokens) else None

        # Build embedding matrix (semantic recall). Best-effort; failure → fallback to pure BM25.
        embed_matrix = None
        if self._embeddings is not None and photos:
            try:
                embed_matrix = await self._build_embedding_matrix(photos)
            except Exception as exc:  # noqa: BLE001
                logger.warning("photo_repo.embed_failed error=%s (fallback to BM25-only)", exc)
                embed_matrix = None

        with self._lock:
            self._photos = photos
            self._bm25 = bm25
            self._embed_matrix = embed_matrix
            self._last_loaded_at = time.time()

        logger.info(
            "photo_repo.loaded count=%d bm25=%s embed=%s sample_tags=%s",
            len(photos),
            "yes" if bm25 is not None else "no",
            "yes" if embed_matrix is not None else "no",
            list(photos[0].tags) if photos else [],
        )

    async def _build_embedding_matrix(self, photos: List[_Photo]) -> Any:
        """Embed each photo's caption+tags into a vector. Returns L2-normalised
        ``numpy.ndarray`` of shape ``(N, D)``. Done in a worker thread because
        ``ZhipuAIEmbeddings.embed_documents`` is blocking I/O.

        Embedding API 限制单次最多 64 条，超过时自动分批。
        """
        import numpy as np  # local import — keep top-level imports lean

        # Build the indexable text per photo. caption_zh is the richest signal;
        # tags are appended as additional tokens so semantic recall is sensitive
        # to them too. Use a separator the embedder won't choke on.
        texts: List[str] = []
        for p in photos:
            parts: List[str] = []
            if p.caption_zh:
                parts.append(p.caption_zh)
            if p.tags:
                parts.append("标签：" + "，".join(p.tags))
            texts.append("。".join(parts) if parts else (p.note or "邓紫棋"))

        # ZhipuAI embedding API 限制单批最多 64 条，超过时自动分批合并
        _BATCH_SIZE = 64
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start:start + _BATCH_SIZE]
            batch_vectors = await asyncio.to_thread(
                self._embeddings.embed_documents, batch
            )
            all_vectors.extend(batch_vectors)
            logger.info(
                "photo_repo.embed_batch progress=%d/%d",
                min(start + _BATCH_SIZE, len(texts)), len(texts),
            )

        mat = np.asarray(all_vectors, dtype="float32")
        if mat.ndim != 2 or mat.shape[0] != len(photos):
            raise RuntimeError(f"unexpected embedding shape: {mat.shape} vs {len(photos)} photos")
        # L2-normalise so cosine = dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        return mat

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` photos best matching the query.

        Ranking pipeline:
            1. BM25 lexical score (tags ×3 / dims ×2 / caption ×1)
            2. Caption embedding cosine score (semantic)
            3. RRF fusion of the two ranked lists
            4. + small VLM-quality tilt
        """
        with self._lock:
            photos = self._photos
            bm25 = self._bm25
            embed_matrix = self._embed_matrix

        if not photos:
            return []

        # ---- BM25 path -------------------------------------------------
        q_tokens = chinese_preprocess_func(query) or [query]
        for t in _HASHTAG_RE.findall(query):
            q_tokens.append(t)
        bm25_scores = bm25.get_scores(q_tokens) if bm25 is not None else None

        # ---- Embedding path -------------------------------------------
        vec_scores = None  # numpy array of cosine similarities, shape (N,)
        if embed_matrix is not None and self._embeddings is not None:
            try:
                vec_scores = self._compute_vec_scores(query, embed_matrix)
            except Exception as exc:  # noqa: BLE001
                logger.warning("photo_repo.query_embed_failed error=%s", exc)
                vec_scores = None

        # ---- Fuse ------------------------------------------------------
        fused = self._fuse_scores(bm25_scores, vec_scores, photos)

        if not fused:
            # Both paths produced nothing meaningful — fallback to top-K by VLM score.
            top = sorted(photos, key=lambda x: x.score, reverse=True)[:top_k]
            return [self._to_hit(p, 0.0) for p in top]

        # Fused is a list of (idx, fused_score) sorted desc.
        out: List[Dict[str, Any]] = []
        for idx, fscore in fused[:top_k]:
            out.append(self._to_hit(photos[idx], float(fscore)))
        return out

    def _compute_vec_scores(self, query: str, embed_matrix: Any) -> Any:
        """Embed query → cosine similarity vector against ``embed_matrix``."""
        import numpy as np

        q_vec = self._embeddings.embed_query(query)
        q = np.asarray(q_vec, dtype="float32")
        if q.ndim != 1 or q.shape[0] != embed_matrix.shape[1]:
            raise RuntimeError(f"query embed shape mismatch: {q.shape} vs matrix {embed_matrix.shape}")
        n = float(np.linalg.norm(q))
        if n == 0:
            return None
        q = q / n
        return embed_matrix @ q  # (N,)

    def _fuse_scores(
        self,
        bm25_scores: Any,
        vec_scores: Any,
        photos: List[_Photo],
    ) -> List[tuple]:
        """RRF fusion of BM25 + embedding rankings, plus a small VLM-quality tilt.

        Returns a list of ``(idx, fused_score)`` sorted by fused_score desc.
        Indices with **zero** signal in both retrievers are dropped (avoids
        returning random pool members for completely off-topic queries).
        """
        n = len(photos)
        # rank dict: photo_idx → 1-based rank in that retriever (lower = more relevant)
        bm25_rank: Dict[int, int] = {}
        vec_rank: Dict[int, int] = {}

        if bm25_scores is not None:
            order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
            for r, i in enumerate(order, start=1):
                if bm25_scores[i] > 0:  # only count actual matches
                    bm25_rank[i] = r

        if vec_scores is not None:
            import numpy as np
            order = list(np.argsort(-vec_scores))  # desc
            for r, i in enumerate(order, start=1):
                # Cosine threshold 0.25: filter out unrelated noise. embedding-3
                # tends to give 0.3+ for genuinely related, 0.15- for unrelated.
                if float(vec_scores[i]) > 0.25:
                    vec_rank[i] = r

        # Union of indices with at least one positive signal
        candidate_idxs = set(bm25_rank.keys()) | set(vec_rank.keys())
        if not candidate_idxs:
            return []

        K = self._rrf_k
        fused: List[tuple] = []
        for i in candidate_idxs:
            rrf = 0.0
            if i in bm25_rank:
                rrf += 1.0 / (K + bm25_rank[i])
            if i in vec_rank:
                rrf += 1.0 / (K + vec_rank[i])
            # Slight quality tilt — ties broken by VLM score
            rrf += self._vlm_score_weight * (photos[i].score / 10.0)
            fused.append((i, rrf))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    @staticmethod
    def _to_hit(p: _Photo, match_score: float) -> Dict[str, Any]:
        return {
            "url": p.url,
            "note": p.note,
            "captionZh": p.caption_zh,
            "tags": list(p.tags),
            "dims": p.dims,
            "score": p.score,
            "match_score": match_score,
        }

    def random_pick(self, k: int = 1) -> List[Dict[str, Any]]:
        """从全池随机抽 k 张。供"用户没明确倾向"的场景使用。

        相比 search()：
        - 不算 BM25 / embedding，零成本
        - 完全均匀随机，新鲜感最强
        - 不要求 query
        """
        import random as _random
        with self._lock:
            photos = list(self._photos)
        if not photos:
            return []
        k = max(1, min(k, len(photos)))
        chosen = _random.sample(photos, k)
        return [self._to_hit(p, 0.0) for p in chosen]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._photos),
                "last_loaded_at": self._last_loaded_at,
                "ttl_seconds": self._ttl,
                "has_bm25": self._bm25 is not None,
                "has_embed": self._embed_matrix is not None,
            }
