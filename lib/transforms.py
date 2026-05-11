"""Dimension-reduction transforms.

Two transforms with the same `.transform(embeddings) -> np.ndarray` interface
so they're swappable in the evaluation loop:

* `TruncateTransform`: MRL-style. Keep the first `target_dim` components.
  Two recipes are supported via `mode`:
    - "standard": truncate then L2-normalize. Use this for our fine-tuned
      MRL model and for any sentence-transformers MatryoshkaLoss output.
    - "nomic": layer_norm -> truncate -> L2-normalize. Required to reproduce
      published nomic-embed-text-v1.5 numbers at sub-768 dims.

* `PCATransform`: post-hoc dimension reduction. Fit on a sample of full-dim
  embeddings, then project to `target_dim`. Output is L2-normalized so it
  can be compared by dot product the same way as truncated embeddings.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Zero rows pass through unchanged."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


class TruncateTransform:
    """MRL-style dimension reduction by keeping the first `target_dim` components."""

    VALID_MODES = ("standard", "nomic")

    def __init__(self, target_dim: int, mode: str = "standard"):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"unknown mode: {mode!r}. valid modes: {self.VALID_MODES}"
            )
        self.target_dim = target_dim
        self.mode = mode

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[1] < self.target_dim:
            raise ValueError(
                f"target_dim={self.target_dim} > input dim {embeddings.shape[1]}"
            )

        x = embeddings
        if self.mode == "nomic":
            t = torch.from_numpy(np.ascontiguousarray(x))
            t = F.layer_norm(t, normalized_shape=(t.shape[1],))
            x = t.numpy()

        x = x[:, : self.target_dim]
        return _l2_normalize(x)

    def __repr__(self) -> str:
        return f"TruncateTransform(target_dim={self.target_dim}, mode={self.mode!r})"


class PCATransform:
    """Post-hoc dimension reduction via PCA, with L2-normalized output."""

    def __init__(self, target_dim: int, random_state: int = 42):
        self.target_dim = target_dim
        self.random_state = random_state
        self._pca: PCA | None = None

    def fit(self, embeddings: np.ndarray) -> "PCATransform":
        if embeddings.shape[1] < self.target_dim:
            raise ValueError(
                f"target_dim={self.target_dim} > input dim {embeddings.shape[1]}"
            )
        self._pca = PCA(
            n_components=self.target_dim,
            random_state=self.random_state,
        )
        self._pca.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("call fit() before transform()")
        projected = self._pca.transform(embeddings)
        return _l2_normalize(projected)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.fit(embeddings).transform(embeddings)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pca": self._pca,
                "target_dim": self.target_dim,
                "random_state": self.random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCATransform":
        data = joblib.load(path)
        obj = cls(
            target_dim=data["target_dim"],
            random_state=data["random_state"],
        )
        obj._pca = data["pca"]
        return obj

    def __repr__(self) -> str:
        state = "fitted" if self._pca is not None else "unfitted"
        return f"PCATransform(target_dim={self.target_dim}, {state})"
