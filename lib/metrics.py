"""Retrieval metrics: nDCG@k, Recall@k, MAP@k.

All metrics take a dense [n_queries, n_docs] score matrix and treat any
qrels score > 0 as relevant. nDCG additionally uses the graded score as
the gain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _build_relevance_map(
    qrels: pd.DataFrame,
    query_ids: list[str],
    doc_ids: list[str],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Build per-query relevance dict and qid->row index lookup.

    Skips qrels rows whose qid or did is not in the supplied id lists,
    or whose score is non-positive.
    """
    qid_to_row = {qid: i for i, qid in enumerate(query_ids)}
    did_set = set(doc_ids)

    rels: dict[str, dict[str, int]] = {}
    for _, r in qrels.iterrows():
        if r.score <= 0:
            continue
        if r.query_id in qid_to_row and r.corpus_id in did_set:
            rels.setdefault(r.query_id, {})[r.corpus_id] = int(r.score)

    return rels, qid_to_row


def _topk_indices(scores_row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores, sorted descending by score."""
    n = scores_row.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, n)
    # argpartition needs an index in [0, n-1]; min(k, n-1) keeps that valid
    part_idx = min(k_eff, n - 1)
    cand = np.argpartition(-scores_row, part_idx)[:k_eff]
    return cand[np.argsort(-scores_row[cand])]


def ndcg_at_k(
    scores: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: pd.DataFrame,
    k: int = 10,
) -> float:
    """Mean nDCG@k over queries with at least one relevant document."""
    rels, qid_to_row = _build_relevance_map(qrels, query_ids, doc_ids)

    ndcgs: list[float] = []
    for qid, rel_map in rels.items():
        i = qid_to_row[qid]
        topk = _topk_indices(scores[i], k)

        dcg = sum(
            rel_map.get(doc_ids[j], 0) / np.log2(rank + 2)
            for rank, j in enumerate(topk)
        )
        ideal = sorted(rel_map.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def recall_at_k(
    scores: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: pd.DataFrame,
    k: int = 100,
) -> float:
    """Mean Recall@k. Counts any qrels score > 0 as relevant."""
    rels, qid_to_row = _build_relevance_map(qrels, query_ids, doc_ids)

    recalls: list[float] = []
    for qid, rel_map in rels.items():
        i = qid_to_row[qid]
        topk = _topk_indices(scores[i], k)

        retrieved = {doc_ids[j] for j in topk}
        relevant = set(rel_map.keys())
        if relevant:
            recalls.append(len(retrieved & relevant) / len(relevant))

    return float(np.mean(recalls)) if recalls else 0.0


def map_at_k(
    scores: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: pd.DataFrame,
    k: int = 10,
) -> float:
    """Mean Average Precision@k. Counts any qrels score > 0 as relevant."""
    rels, qid_to_row = _build_relevance_map(qrels, query_ids, doc_ids)

    aps: list[float] = []
    for qid, rel_map in rels.items():
        i = qid_to_row[qid]
        topk = _topk_indices(scores[i], k)

        relevant = set(rel_map.keys())
        if not relevant:
            continue

        hits = 0
        sum_precisions = 0.0
        for rank, j in enumerate(topk):
            if doc_ids[j] in relevant:
                hits += 1
                sum_precisions += hits / (rank + 1)

        aps.append(sum_precisions / min(len(relevant), k))

    return float(np.mean(aps)) if aps else 0.0


def evaluate_run(
    scores: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: pd.DataFrame,
) -> dict[str, float]:
    """Compute all standard metrics in one pass and return as a dict."""
    return {
        "ndcg_at_10": ndcg_at_k(scores, query_ids, doc_ids, qrels, k=10),
        "recall_at_100": recall_at_k(scores, query_ids, doc_ids, qrels, k=100),
        "map_at_10": map_at_k(scores, query_ids, doc_ids, qrels, k=10),
    }
