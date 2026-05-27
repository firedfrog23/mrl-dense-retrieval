from pathlib import Path

from lib.training import run_training

ROOT = Path(__file__).parent.resolve()


if __name__ == "__main__":
    run_training(
        config_path=ROOT / "configs" / "train_baseline.yaml",
        base_config_path=ROOT / "configs" / "base.yaml",
        project_root=ROOT,
    )