"""Run B: fine-tune bge-base on MS MARCO triplets with MatryoshkaLoss.

Wraps MultipleNegativesRankingLoss in MatryoshkaLoss with nested dims
[768, 512, 256, 128, 64, 32]. Same data, seed, batch size, lr, and epochs
as Run A; only the loss differs. This is the MRL contribution.

Hyperparameters are in configs/train_mrl.yaml (overrides) and
configs/base.yaml (shared defaults).
"""
from pathlib import Path

from lib.training import run_training

ROOT = Path(__file__).parent.resolve()


if __name__ == "__main__":
    run_training(
        config_path=ROOT / "configs" / "train_mrl.yaml",
        base_config_path=ROOT / "configs" / "base.yaml",
        project_root=ROOT,
    )