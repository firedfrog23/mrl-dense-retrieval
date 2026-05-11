"""Run A: fine-tune bge-base on MS MARCO triplets with MultipleNegativesRankingLoss.

This is the non-MRL baseline. PCA gets fitted on its outputs in Phase 3
to give a fair head-to-head comparison against Run B at sub-768 dims.

Hyperparameters are in configs/train_baseline.yaml (overrides) and
configs/base.yaml (shared defaults).
"""
from pathlib import Path

from lib.training import run_training

ROOT = Path(__file__).parent.resolve()


if __name__ == "__main__":
    run_training(
        config_path=ROOT / "configs" / "train_baseline.yaml",
        base_config_path=ROOT / "configs" / "base.yaml",
        project_root=ROOT,
    )