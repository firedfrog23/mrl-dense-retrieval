import logging
from pathlib import Path

import pandas as pd
import torch

from lib import load_config, setup_logging
from lib.beir_loader import load_beir
from lib.evaluation import build_transform, get_or_compute_embeddings
from lib.logging_utils import relpath_str
from lib.metrics import evaluate_run

ROOT = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)


def _save_results(results: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return len(df)


def main() -> None:
    cfg = load_config(
        ROOT / "configs" / "eval.yaml",
        base_path=ROOT / "configs" / "base.yaml",
    )
    setup_logging(log_file=ROOT / cfg["log_file"])

    logger.info("=== Phase 4: evaluation matrix ===")

    variants = cfg["variants"]
    beir_tasks = cfg["eval"]["beir_tasks"]
    dims_to_eval = cfg["eval"]["dims_to_eval"]
    cache_dir = ROOT / cfg["eval"]["embeddings_cache_dir"]
    pca_dir = ROOT / "outputs" / "pca_models"
    out_csv = ROOT / cfg["results_csv"]

    n_cells = len(variants) * len(beir_tasks) * len(dims_to_eval)
    logger.info(f"variants:  {[v['name'] for v in variants]}")
    logger.info(f"datasets:  {beir_tasks}")
    logger.info(f"dims:      {dims_to_eval}")
    logger.info(f"cells:     {n_cells}")
    logger.info(f"cache dir: {relpath_str(cache_dir, ROOT)}")
    logger.info(f"output:    {relpath_str(out_csv, ROOT)}")

    results: list[dict] = []

    for ds_idx, dataset_name in enumerate(beir_tasks, 1):
        logger.info("")
        logger.info(f"[{ds_idx}/{len(beir_tasks)}] dataset: {dataset_name}")

        ds = load_beir(
            dataset_name,
            datasets_dir=ROOT / "datasets",
            qrels_split=cfg["eval"]["qrels_split"],
        )
        logger.info(f"  loaded: {ds}")

        for variant in variants:
            logger.info(f"  variant: {variant['name']}")

            corpus_emb, query_emb = get_or_compute_embeddings(
                encoder_path=ROOT / variant["encoder_path"],
                encoder_family=variant["encoder_family"],
                dataset_name=dataset_name,
                dataset=ds,
                cache_dir=cache_dir,
            )
            full_dim = corpus_emb.shape[1]

            for dim in dims_to_eval:
                if dim > full_dim:
                    logger.warning(f"    skipping dim={dim} (> full_dim={full_dim})")
                    continue

                transform = build_transform(
                    variant=variant,
                    target_dim=dim,
                    full_dim=full_dim,
                    pca_dir=pca_dir,
                )

                c_t = transform(corpus_emb)
                q_t = transform(query_emb)

                scores = q_t @ c_t.T
                metrics = evaluate_run(
                    scores=scores,
                    query_ids=ds.query_ids,
                    doc_ids=ds.corpus_ids,
                    qrels=ds.qrels,
                )

                results.append({
                    "variant": variant["name"],
                    "dim": dim,
                    "dataset": dataset_name,
                    **metrics,
                })

                logger.info(
                    f"    dim={dim:>3}: "
                    f"ndcg@10={metrics['ndcg_at_10']:.4f}  "
                    f"recall@100={metrics['recall_at_100']:.4f}  "
                    f"map@10={metrics['map_at_10']:.4f}"
                )

        # Free dataset memory and clear GPU cache between datasets.
        # TREC-COVID corpus is 526 MB at full dim; keeping prior datasets
        # around blows up RAM on the next encoding pass.
        del ds, corpus_emb, query_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Checkpoint after each dataset. If something crashes on the next
        # one, we keep all completed results.
        n_rows = _save_results(results, out_csv)
        logger.info(f"  checkpointed {n_rows} rows -> {out_csv.name}")

    logger.info("")
    logger.info(f"=== done. {len(results)} rows -> {relpath_str(out_csv, ROOT)} ===")
    logger.info("summary by variant (avg nDCG@10 across all dims and datasets):")
    df = pd.DataFrame(results)
    summary = df.groupby("variant")["ndcg_at_10"].mean().sort_values(ascending=False)
    for variant, val in summary.items():
        logger.info(f"  {variant:>20}: {val:.4f}")


if __name__ == "__main__":
    main()
