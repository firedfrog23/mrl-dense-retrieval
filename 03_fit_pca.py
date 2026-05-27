import json
import logging
from pathlib import Path

from lib import load_config, setup_logging
from lib.encoder import make_encoder
from lib.pca_fitting import fit_pcas, load_unique_passages

ROOT = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config(
        ROOT / "configs" / "eval.yaml",
        base_path=ROOT / "configs" / "base.yaml",
    )
    setup_logging(log_file=ROOT / "outputs" / "logs" / "fit_pca.log")

    logger.info("=== Phase 3: fit PCA on Run A ===")

    runa_path = ROOT / "checkpoints" / "bge-runA"
    if not runa_path.exists():
        raise FileNotFoundError(
            f"Run A checkpoint not found: {runa_path}. "
            f"Run Phase 2 (02_train_baseline.py) first."
        )

    logger.info(f"loading Run A from {runa_path.name}")
    enc = make_encoder(model_path=runa_path, family="bge")

    n_sample = cfg["eval"]["pca_fit_sample_size"]
    triplets_path = ROOT / cfg["training"]["triplets_parquet"]
    logger.info(f"sampling {n_sample:,} unique passages from {triplets_path.name}")
    passages = load_unique_passages(triplets_path, n=n_sample, seed=cfg["seed"])
    logger.info(f"got {len(passages):,} passages")

    logger.info(f"encoding at full {enc.full_dim} dim ...")
    embeddings = enc.encode_corpus(passages, show_progress=True)
    logger.info(f"embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    target_dims = [d for d in cfg["eval"]["dims_to_eval"] if d < enc.full_dim]
    logger.info(f"fitting PCAs for dims: {target_dims}")
    pcas = fit_pcas(embeddings, target_dims=target_dims, random_state=cfg["seed"])

    out_dir = ROOT / "outputs" / "pca_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d, pca in pcas.items():
        path = out_dir / f"pca_{d}.joblib"
        pca.save(path)
        logger.info(f"  saved {path.name}")

    metadata = {
        "encoder_path": str(runa_path.relative_to(ROOT)).replace("\\", "/"),
        "encoder_family": enc.family,
        "full_dim": enc.full_dim,
        "fit_sample_size": len(passages),
        "seed": cfg["seed"],
        "target_dims": target_dims,
        "explained_variance_ratio_sum": {
            d: float(p._pca.explained_variance_ratio_.sum())
            for d, p in pcas.items()
        },
    }
    info_path = out_dir / "fit_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"saved metadata: {info_path.name}")

    logger.info("=== Phase 3 done ===")
    logger.info("explained variance summary:")
    for d in target_dims:
        ev = metadata["explained_variance_ratio_sum"][d]
        logger.info(f"  dim={d:>3}: {ev:.4f}")


if __name__ == "__main__":
    main()
