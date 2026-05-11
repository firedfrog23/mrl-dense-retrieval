"""YAML config loader with deep-merge support.

Pattern: every script loads `configs/base.yaml` and overlays a phase-specific
config. Override values win on conflict. Nested dicts are merged recursively
rather than replaced wholesale.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. `override` wins on conflicts."""
    out = deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def load_config(
    config_path: str | Path,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML config, optionally overlaying it onto a base config.

    Parameters
    ----------
    config_path : str | Path
        Path to the override config (e.g. ``configs/train_mrl.yaml``).
    base_path : str | Path | None
        Optional path to a base config (e.g. ``configs/base.yaml``).
        When given, `config_path` is deep-merged on top of `base_path`.

    Returns
    -------
    dict
        Merged config.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if base_path is None:
        return cfg

    base_path = Path(base_path)
    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    return _deep_merge(base, cfg)
