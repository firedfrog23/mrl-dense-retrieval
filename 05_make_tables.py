import logging
from pathlib import Path

from lib import setup_logging
from lib.analysis import (
    load_results,
    write_main_results_table,
    write_mrl_vs_pca_table,
    write_per_dataset_table,
)

ROOT = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    csv_path = ROOT / "outputs" / "results" / "eval_results.csv"
    tables_dir = ROOT / "outputs" / "tables"

    logger.info(f"loading {csv_path}")
    df = load_results(csv_path)
    logger.info(
        f"  {len(df)} rows, "
        f"{df['variant'].nunique()} variants, "
        f"{df['dim'].nunique()} dims, "
        f"{df['dataset'].nunique()} datasets"
    )

    main_path = tables_dir / "main_results.tex"
    write_main_results_table(df, main_path)
    logger.info(f"wrote {main_path.relative_to(ROOT)}")

    per_path = tables_dir / "per_dataset_at_dim32.tex"
    write_per_dataset_table(df, per_path, dim=32)
    logger.info(f"wrote {per_path.relative_to(ROOT)}")

    vs_path = tables_dir / "mrl_vs_pca.tex"
    write_mrl_vs_pca_table(df, vs_path)
    logger.info(f"wrote {vs_path.relative_to(ROOT)}")

    logger.info("=== done ===")


if __name__ == "__main__":
    main()
