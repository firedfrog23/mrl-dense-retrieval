"""Shared library for the MRL retrieval study.

Public surface re-exported for convenience.
"""
from lib.beir_loader import BeirDataset, load_beir
from lib.config import load_config
from lib.encoder import RetrievalEncoder, make_encoder
from lib.logging_utils import setup_logging
from lib.metrics import evaluate_run, map_at_k, ndcg_at_k, recall_at_k
from lib.transforms import PCATransform, TruncateTransform

__all__ = [
    "BeirDataset", "load_beir",
    "load_config",
    "RetrievalEncoder", "make_encoder",
    "setup_logging",
    "evaluate_run", "map_at_k", "ndcg_at_k", "recall_at_k",
    "PCATransform", "TruncateTransform",
]
