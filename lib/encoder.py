"""Encoder wrapper that hides model-specific quirks.

Different retrieval models expect different input formatting:

* bge family prepends an instruction to QUERIES only.
* nomic-embed-text-v1.5 prepends "search_query:" to queries and
  "search_document:" to documents. Forgetting these silently tanks recall.
* all-mpnet-base-v2 needs no prefix.

This module detects the family from the folder name and applies the right
prefix automatically. Outputs are L2-normalized so dot product == cosine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# Per-family prefix conventions. Empty string means "no prefix".
PREFIX_RULES: dict[str, dict[str, str]] = {
    "bge": {
        "query": "Represent this sentence for searching relevant passages: ",
        "doc": "",
    },
    "nomic": {
        "query": "search_query: ",
        "doc": "search_document: ",
    },
    "mpnet": {"query": "", "doc": ""},
    "default": {"query": "", "doc": ""},
}


def _detect_model_family(model_path: str) -> str:
    """Identify the model family from the folder or repo name."""
    name = Path(model_path).name.lower()
    if "bge" in name:
        return "bge"
    if "nomic" in name:
        return "nomic"
    if "mpnet" in name:
        return "mpnet"
    return "default"


class RetrievalEncoder:
    """A SentenceTransformer wrapper for retrieval.

    Responsibilities:
        * apply family-specific query/doc prefixes
        * L2-normalize output vectors
        * expose `encode_queries` and `encode_corpus` so callers don't
          have to remember which prefix goes where
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        family: str | None = None,
        max_seq_length: int | None = None,
        batch_size: int = 128,
    ):
        self.model_path = str(model_path)
        self.family = family or _detect_model_family(self.model_path)
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        kwargs: dict = {"device": device}
        if self.family == "nomic":
            # Nomic ships custom modeling code that needs trust_remote_code=True
            # on older transformers releases. Harmless on newer ones.
            kwargs["trust_remote_code"] = True

        self.model = SentenceTransformer(self.model_path, **kwargs)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

        prefixes = PREFIX_RULES.get(self.family, PREFIX_RULES["default"])
        self.query_prefix = prefixes["query"]
        self.doc_prefix = prefixes["doc"]
        self.full_dim = self.model.get_sentence_embedding_dimension()

    def _encode(
        self,
        texts: list[str],
        prefix: str,
        show_progress: bool,
    ) -> np.ndarray:
        prefixed = [prefix + t for t in texts] if prefix else texts
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_queries(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode queries to L2-normalized vectors. Shape: [n, full_dim]."""
        return self._encode(texts, self.query_prefix, show_progress)

    def encode_corpus(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode documents to L2-normalized vectors. Shape: [n, full_dim]."""
        return self._encode(texts, self.doc_prefix, show_progress)

    def __repr__(self) -> str:
        return (
            f"RetrievalEncoder(family={self.family!r}, "
            f"path={Path(self.model_path).name!r}, "
            f"dim={self.full_dim}, device={self.device!r})"
        )


def make_encoder(model_path: str | Path, **kwargs) -> RetrievalEncoder:
    """Factory: build a RetrievalEncoder with sensible defaults."""
    return RetrievalEncoder(model_path=model_path, **kwargs)
