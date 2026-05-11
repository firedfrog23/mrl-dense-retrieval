"""Evaluation pipeline utilities for Phase 4.

Two responsibilities:

* `get_or_compute_embeddings` caches per-(encoder, dataset) full-dim
  embeddings to disk as .npy files. First call encodes; later calls load.
  This is what makes the 120-cell eval matrix tractable: each unique
  (encoder, dataset) pair is encoded at most once.

* `build_transform` returns a callable `(embeddings) -> embeddings` that
  applies the variant's truncation or PCA at the target dim. At dim==full_dim
  for the PCA variant, returns identity (we don't fit a PCA at full dim).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from lib.beir_loader import BeirDataset
from lib.encoder import make_encoder
from lib.transforms import PCATransform, TruncateTransform

logger = logging.getLogger(__name__)

EmbeddingTransform = Callable[[np.ndarray], np.ndarray]


# Per-family encode-time batch size. Nomic uses ~4x more activation memory
# per token than bge-base because of its larger SwiGLU MLPs (4x intermediate
# dim), so batch=128 OOMs on 16 GB cards when corpora have longer texts
# (e.g. FiQA financial passages). batch=32 works comfortably.
ENCODE_BATCH_SIZE = {
    "bge": 128,
    "mpnet": 128,
    "nomic": 32,
    "default": 64,
}


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def get_or_compute_embeddings(
    encoder_path: str | Path,
    encoder_family: str,
    dataset_name: str,
    dataset: BeirDataset,
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corpus_emb, query_emb) at full dim. Caches on first call.

    Cache layout:
        {cache_dir}/{encoder_name}/{dataset_name}_corpus.npy
        {cache_dir}/{encoder_name}/{dataset_name}_queries.npy

    `encoder_name` is the last path component of encoder_path. If you swap
    a model under the same folder name, delete the cache to avoid stale
    embeddings.
    """
    encoder_path = Path(encoder_path)
    cache_dir = Path(cache_dir)
    enc_name = encoder_path.name

    corpus_path = cache_dir / enc_name / f"{dataset_name}_corpus.npy"
    queries_path = cache_dir / enc_name / f"{dataset_name}_queries.npy"

    if corpus_path.exists() and queries_path.exists():
        logger.info(f"  cache hit: {enc_name}/{dataset_name}")
        return np.load(corpus_path), np.load(queries_path)

    batch_size = ENCODE_BATCH_SIZE.get(encoder_family, ENCODE_BATCH_SIZE["default"])
    logger.info(
        f"  encoding {dataset_name} with {enc_name} (family={encoder_family}, batch={batch_size})"
    )
    enc = make_encoder(
        model_path=encoder_path,
        family=encoder_family,
        batch_size=batch_size,
    )
    corpus_emb = enc.encode_corpus(dataset.corpus_text, show_progress=True)
    query_emb = enc.encode_queries(dataset.query_text, show_progress=False)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(corpus_path, corpus_emb)
    np.save(queries_path, query_emb)
    logger.info(f"  cached: {enc_name}/{dataset_name}")

    # Free the encoder. Phase 4 cycles through 3+ encoders; not freeing each
    # one bloats GPU memory and increases the chance of the next encoder OOM.
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return corpus_emb, query_emb


def build_transform(
    variant: dict,
    target_dim: int,
    full_dim: int,
    pca_dir: str | Path,
) -> EmbeddingTransform:
    """Build the embeddings transform for one cell of the eval matrix.

    Returned callable takes [n, full_dim] and returns [n, target_dim].
    For the pca variant at dim==full_dim, returns identity (no PCA fitted there).
    """
    if variant["transform"] == "truncate":
        return TruncateTransform(
            target_dim=target_dim,
            mode=variant["transform_mode"],
        ).transform

    if variant["transform"] == "pca":
        if target_dim >= full_dim:
            return _identity
        pca_path = Path(pca_dir) / f"pca_{target_dim}.joblib"
        if not pca_path.exists():
            raise FileNotFoundError(
                f"PCA file not found: {pca_path.name}. Run Phase 3 (04_fit_pca.py) first."
            )
        return PCATransform.load(pca_path).transform

    raise ValueError(f"unknown transform type: {variant['transform']!r}")
