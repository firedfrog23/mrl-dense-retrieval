"""Training utilities shared between Run A and Run B.

Run A: standard fine-tune with MultipleNegativesRankingLoss.
Run B: same, but wrapped in MatryoshkaLoss for nested-dim training.

Both runs use the exact same data, seed, batch size, lr, and epoch count.
Only the loss differs.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from torch.utils.data import DataLoader

from lib.beir_loader import load_beir
from lib.config import load_config
from lib.encoder import make_encoder
from lib.logging_utils import setup_logging
from lib.metrics import ndcg_at_k

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_triplets(
    parquet_path: str | Path,
    num_samples: int | None = None,
    seed: int = 42,
) -> list[InputExample]:
    """Load (query, positive, negative) triplets from a parquet file.

    Subsamples to `num_samples` rows with a fixed seed if given.
    """
    df = pd.read_parquet(parquet_path)
    expected = {"query", "positive", "negative"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"parquet missing columns: {missing}. found: {sorted(df.columns)}"
        )

    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)

    return [
        InputExample(texts=[row["query"], row["positive"], row["negative"]])
        for _, row in df.iterrows()
    ]


def build_loss(
    model: SentenceTransformer,
    loss_type: str,
    mrl_dims: list[int] | None = None,
    mrl_weights: list[float] | None = None,
) -> torch.nn.Module:
    """Build the training loss.

    loss_type:
        "mnrl" - MultipleNegativesRankingLoss (Run A)
        "mrl"  - MatryoshkaLoss wrapping MNRL (Run B)
    """
    inner = MultipleNegativesRankingLoss(model)

    if loss_type == "mnrl":
        return inner

    if loss_type == "mrl":
        if not mrl_dims:
            raise ValueError("mrl loss requires mrl_dims")
        return MatryoshkaLoss(
            model,
            inner,
            matryoshka_dims=mrl_dims,
            matryoshka_weights=mrl_weights,
        )

    raise ValueError(
        f"unknown loss_type: {loss_type!r}. valid: 'mnrl', 'mrl'"
    )


def train(
    model: SentenceTransformer,
    examples: list[InputExample],
    loss: torch.nn.Module,
    training_cfg: dict,
    output_path: str | Path,
) -> None:
    """Run fine-tuning. Saves the trained model to `output_path`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_workers = training_cfg.get("num_workers", 0)
    if sys.platform == "win32" and num_workers > 0:
        logger.warning(
            f"Windows: forcing num_workers=0 (was {num_workers}) to avoid "
            "DataLoader multiprocessing issues"
        )
        num_workers = 0

    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=training_cfg["batch_size"],
        num_workers=num_workers,
    )

    logger.info("starting training")
    logger.info(f"  examples:    {len(examples):,}")
    logger.info(f"  batch_size:  {training_cfg['batch_size']}")
    logger.info(f"  steps/epoch: {len(loader):,}")
    logger.info(f"  epochs:      {training_cfg['epochs']}")
    logger.info(f"  lr:          {training_cfg['learning_rate']}")
    logger.info(f"  warmup:      {training_cfg['warmup_steps']}")
    logger.info(f"  fp16:        {training_cfg.get('fp16', False)}")
    logger.info(f"  output:      {output_path.name}")

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=training_cfg["epochs"],
        warmup_steps=training_cfg["warmup_steps"],
        optimizer_params={"lr": training_cfg["learning_rate"]},
        output_path=str(output_path),
        use_amp=training_cfg.get("fp16", False),
        show_progress_bar=True,
    )
    logger.info("training done")


def quick_eval_nfcorpus(
    model_path: str | Path,
    datasets_dir: str | Path,
    family: str = "bge",
) -> float:
    """Encode NFCorpus and return nDCG@10. Quick post-training sanity check."""
    enc = make_encoder(model_path=model_path, family=family)
    ds = load_beir("nfcorpus", datasets_dir=datasets_dir)

    corpus_emb = enc.encode_corpus(ds.corpus_text, show_progress=False)
    query_emb = enc.encode_queries(ds.query_text, show_progress=False)
    scores = query_emb @ corpus_emb.T

    return ndcg_at_k(scores, ds.query_ids, ds.corpus_ids, ds.qrels, k=10)


def run_training(
    config_path: str | Path,
    base_config_path: str | Path,
    project_root: str | Path,
) -> float:
    """End-to-end training entry. Loads config, trains, returns NFCorpus nDCG@10.

    The two Phase 2 entry-point scripts (02_train_baseline.py, 03_train_mrl.py)
    are thin wrappers around this function.
    """
    project_root = Path(project_root)
    cfg = load_config(config_path, base_path=base_config_path)
    setup_logging(log_file=project_root / cfg["log_file"])

    logger.info(f"=== {cfg['run_name']} ===")
    seed_everything(cfg["seed"])

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; training on CPU will be very slow")

    base_path = project_root / cfg["base_model"]["path"]
    logger.info(f"loading base model: {base_path.name}")
    model = SentenceTransformer(str(base_path))
    model.max_seq_length = cfg["base_model"]["max_seq_length"]

    triplets_path = project_root / cfg["training"]["triplets_parquet"]
    logger.info(f"loading triplets: {triplets_path.name}")
    examples = load_triplets(
        triplets_path,
        num_samples=cfg["training"]["num_train_samples"],
        seed=cfg["seed"],
    )

    mrl_cfg = cfg.get("mrl", {})
    loss = build_loss(
        model,
        loss_type=cfg["loss"],
        mrl_dims=mrl_cfg.get("dims"),
        mrl_weights=mrl_cfg.get("weights"),
    )
    logger.info(f"loss: {type(loss).__name__}")

    output_path = project_root / cfg["checkpoint_dir"]
    train(model, examples, loss, cfg["training"], output_path)

    logger.info("running quick NFCorpus eval ...")
    ndcg = quick_eval_nfcorpus(
        model_path=output_path,
        datasets_dir=project_root / "datasets",
        family=cfg["base_model"]["family"],
    )
    logger.info(f"NFCorpus nDCG@10 after {cfg['run_name']}: {ndcg:.4f}")
    logger.info("(zero-shot bge-base baseline was 0.3739; expect roughly that)")
    return ndcg
