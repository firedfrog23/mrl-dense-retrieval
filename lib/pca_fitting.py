"""PCA fitting helpers for Phase 3.

The actual PCA math lives in `lib.transforms.PCATransform`. This module
adds the small bits of glue that are specific to fitting PCA on MS MARCO
training data:

  * `load_unique_passages` samples unique positive passages from the
    triplets parquet so we don't overweight passages that appear as the
    positive for many queries.
  * `fit_pcas` fits one `PCATransform` per target dim against the same
    training embeddings, with diagnostic logging of explained variance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lib.transforms import PCATransform

logger = logging.getLogger(__name__)


def load_unique_passages(
    triplets_parquet: str | Path,
    n: int,
    seed: int = 42,
) -> list[str]:
    """Sample up to `n` unique positive passages from the triplets file.

    Reads only the `positive` column from the parquet to keep memory low.
    If fewer than `n` unique passages are available, returns all of them
    and logs a warning.
    """
    df = pd.read_parquet(triplets_parquet, columns=["positive"])
    unique = df["positive"].drop_duplicates()
    if len(unique) < n:
        logger.warning(
            f"only {len(unique):,} unique passages available, requested {n:,}"
        )
        return unique.tolist()
    return unique.sample(n=n, random_state=seed).tolist()


def fit_pcas(
    embeddings: np.ndarray,
    target_dims: list[int],
    random_state: int = 42,
) -> dict[int, PCATransform]:
    """Fit one PCATransform per target dim. All share the same training data.

    Returns a dict mapping target_dim -> fitted PCATransform.
    """
    pcas: dict[int, PCATransform] = {}
    for d in target_dims:
        logger.info(f"  fitting PCA for dim={d}")
        pca = PCATransform(target_dim=d, random_state=random_state)
        pca.fit(embeddings)
        pcas[d] = pca
        ev = float(pca._pca.explained_variance_ratio_.sum())
        logger.info(f"    explained variance: {ev:.4f}")
    return pcas
