"""SAGE graph builder — constructs RELATED_TO edges for cross-chunk expansion.

Prototype-safe design:
  - Uses top-K-per-chunk candidate selection (not full pairwise O(n²))
  - Applies a hard minimum score floor (SAGE_MIN_SCORE) before percentile pruning
  - Caps total edges written via SAGE_MAX_EDGES
  - Skips self-loops and reciprocal duplicates

Two edge types:
  RELATED_TO {method: "sage_semantic"}  — based on chunk embedding similarity
  RELATED_TO {method: "sage_metadata"}  — based on metadata string similarity

For prototype runs (--limit 1000 → 2000 chunks), default settings produce
roughly 200–800 RELATED_TO edges — a thin, high-confidence expansion layer.

Scale guide:
  ≤ 10k chunks  : exact mode works fine (set SAGE_TOP_K_PER_CHUNK=50)
  > 10k chunks  : install faiss-cpu for ANN — falls back to numpy if absent
"""
from __future__ import annotations

import logging
import math
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from etl.chunker_v3 import ChunkData

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ── Similarity helpers ─────────────────────────────────────────────────────────

def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    """Return n×n cosine similarity matrix for a batch of L2-normalised vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = vecs / norms
    return normed @ normed.T


def _top_k_pairs(
    sim: np.ndarray,
    chunk_ids: list[str],
    k: int,
    min_score: float,
    method: str,
) -> list[dict]:
    """Extract top-k neighbours per row; return edge dicts, deduplicated."""
    n = sim.shape[0]
    # Mask diagonal (self) and below-floor entries
    np.fill_diagonal(sim, 0.0)
    sim[sim < min_score] = 0.0

    edges: dict[tuple[str, str], dict] = {}
    for i in range(n):
        row = sim[i]
        top_idx = np.argpartition(row, max(-k, -n))[-k:]
        for j in top_idx:
            score = float(row[j])
            if score <= 0:
                continue
            a, b = chunk_ids[i], chunk_ids[j]
            key = (min(a, b), max(a, b))   # canonical order — no duplicates
            if key not in edges or score > edges[key]["score"]:
                edges[key] = {"src_id": a, "dst_id": b, "score": score, "method": method}

    return list(edges.values())


# ── FAISS ANN (optional, large-scale) ─────────────────────────────────────────

def _ann_top_k(vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) arrays of shape (n, k) using FAISS if available."""
    try:
        import faiss  # noqa: PLC0415
        d = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = (vecs / np.where(norms == 0, 1e-9, norms)).astype("float32")
        index = faiss.IndexFlatIP(d)
        index.add(normed)
        scores, indices = index.search(normed, k + 1)   # +1 to exclude self
        return indices[:, 1:], scores[:, 1:]            # drop self (col 0)
    except ImportError:
        log.debug("faiss not installed — falling back to numpy top-k")
        sim = _cosine_matrix(vecs)
        n = sim.shape[0]
        np.fill_diagonal(sim, -1)
        indices = np.argsort(-sim, axis=1)[:, :k]
        scores  = np.take_along_axis(sim, indices, axis=1)
        return indices, scores


def _ann_pairs(
    vecs: np.ndarray,
    chunk_ids: list[str],
    k: int,
    min_score: float,
    percentile: float,
    method: str,
) -> list[dict]:
    """Build edges using ANN top-k + percentile threshold."""
    indices, scores = _ann_top_k(vecs, k)

    # Compute threshold from the score distribution of all candidates
    all_scores = scores.flatten()
    all_scores = all_scores[all_scores > min_score]
    if len(all_scores) == 0:
        return []
    threshold = float(np.percentile(all_scores, percentile))
    threshold = max(threshold, min_score)
    log.info("  [SAGE %s] percentile=%.0f → threshold=%.4f  (candidates: %d)",
             method, percentile, threshold, len(all_scores))

    edges: dict[tuple[str, str], dict] = {}
    for i, (nbr_row, score_row) in enumerate(zip(indices, scores)):
        for j, score in zip(nbr_row, score_row):
            if score < threshold:
                continue
            a, b = chunk_ids[i], chunk_ids[int(j)]
            key = (min(a, b), max(a, b))
            if key not in edges or float(score) > edges[key]["score"]:
                edges[key] = {"src_id": a, "dst_id": b, "score": float(score), "method": method}

    return list(edges.values())


# ── Metadata string embedding ──────────────────────────────────────────────────

def _metadata_strings(chunks: list[ChunkData]) -> list[str]:
    """Extract the context_prefix (metadata summary) from each chunk for meta-similarity."""
    return [c.context_prefix if c.context_prefix else c.chunk_type for c in chunks]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_sage_edges(chunks: list[ChunkData]) -> list[dict]:
    """Compute SAGE RELATED_TO edge dicts from chunk embeddings + metadata strings.

    Returns list of dicts: {src_id, dst_id, score, method}
    """
    if not config.SAGE_ENABLED:
        log.info("SAGE disabled (SAGE_ENABLED=false)")
        return []

    n = len(chunks)
    if n < 4:
        log.info("SAGE skipped — too few chunks (%d)", n)
        return []

    # Verify all chunks have embeddings
    embedded = [c for c in chunks if c.embedding is not None]
    if len(embedded) < n:
        log.warning("SAGE: %d/%d chunks have embeddings — using embedded subset", len(embedded), n)
        chunks = embedded
        n = len(chunks)

    k          = min(config.SAGE_TOP_K_PER_CHUNK, n - 1)
    min_score  = config.SAGE_MIN_SCORE
    percentile = config.SAGE_PERCENTILE
    max_edges  = config.SAGE_MAX_EDGES

    log.info("Building SAGE edges: %d chunks  k=%d  min_score=%.2f  max_edges=%d",
             n, k, min_score, max_edges)

    chunk_ids = [c.id for c in chunks]

    # ── Phase 1: semantic similarity (chunk embeddings) ────────────────────────
    sem_vecs = np.array([c.embedding for c in chunks], dtype=np.float32)
    sem_edges = _ann_pairs(sem_vecs, chunk_ids, k, min_score, percentile, "sage_semantic")
    log.info("  [SAGE semantic] %d candidate edges", len(sem_edges))

    # ── Phase 2: metadata similarity (context_prefix strings) ─────────────────
    meta_strings = _metadata_strings(chunks)
    from etl.embedder import embed_texts  # noqa: PLC0415
    log.info("  [SAGE metadata] embedding %d metadata strings...", n)
    meta_embs = embed_texts(meta_strings)
    meta_vecs = np.array(meta_embs, dtype=np.float32)
    meta_edges = _ann_pairs(meta_vecs, chunk_ids, k, min_score, percentile, "sage_metadata")
    log.info("  [SAGE metadata] %d candidate edges", len(meta_edges))

    # ── Merge and cap ──────────────────────────────────────────────────────────
    # Deduplicate across both phases (keep higher score, prefer semantic)
    all_edges: dict[tuple[str, str], dict] = {}
    for edge in sem_edges + meta_edges:
        key = (min(edge["src_id"], edge["dst_id"]), max(edge["src_id"], edge["dst_id"]))
        if key not in all_edges or edge["score"] > all_edges[key]["score"]:
            all_edges[key] = edge

    merged = sorted(all_edges.values(), key=lambda e: e["score"], reverse=True)

    # Safety cap for prototype graph size
    if len(merged) > max_edges:
        log.warning("SAGE: capping %d edges → %d (SAGE_MAX_EDGES limit)", len(merged), max_edges)
        merged = merged[:max_edges]

    log.info("SAGE: writing %d RELATED_TO edges  (semantic: %d  metadata: %d)",
             len(merged),
             sum(1 for e in merged if e["method"] == "sage_semantic"),
             sum(1 for e in merged if e["method"] == "sage_metadata"))

    return merged
