"""Phase 5b: generate PDF figures from outputs/results/eval_results.csv.

Writes three figures to outputs/figures/, sized for ACL single column
(3.3 inches wide), serif font, editable PDF text:
    ndcg_vs_dim.pdf            avg nDCG@10 vs dim, one line per variant
    per_dataset_heatmap.pdf    dataset x dim retention heatmap (Run B)
    mrl_vs_pca_gap.pdf         per-dim bar chart of MRL minus PCA

Pulled into the paper directly:
    \\begin{figure}
      \\centering
      \\includegraphics[width=\\columnwidth]{figures/ndcg_vs_dim.pdf}
      \\caption{...}
    \\end{figure}
"""
import logging
from pathlib import Path

from lib import setup_logging
from lib.analysis import (
    load_results,
    plot_mrl_vs_pca_gap,
    plot_ndcg_vs_dim,
    plot_per_dataset_heatmap,
)

ROOT = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    csv_path = ROOT / "outputs" / "results" / "eval_results.csv"
    figures_dir = ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading {csv_path}")
    df = load_results(csv_path)
    logger.info(f"  {len(df)} rows")

    p1 = figures_dir / "ndcg_vs_dim.pdf"
    plot_ndcg_vs_dim(df, p1)
    logger.info(f"wrote {p1.relative_to(ROOT)}")

    p2 = figures_dir / "per_dataset_heatmap.pdf"
    plot_per_dataset_heatmap(df, p2, variant="runB-truncate")
    logger.info(f"wrote {p2.relative_to(ROOT)} (variant: runB-truncate)")

    p3 = figures_dir / "mrl_vs_pca_gap.pdf"
    plot_mrl_vs_pca_gap(df, p3)
    logger.info(f"wrote {p3.relative_to(ROOT)}")

    logger.info("=== done ===")


if __name__ == "__main__":
    main()
