"""Shared logger configuration.

Every script calls `setup_logging(...)` once at startup. After that, any
`logging.getLogger(__name__)` call returns a logger that writes to both
stdout and (optionally) a file with a consistent format.

The format intentionally omits timestamps and absolute paths so logs are
safe to commit or share without leaking machine details.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FMT = "%(levelname)-7s | %(name)s | %(message)s"


# Per-library log levels. Anything above WARNING goes to a logger's local
# threshold; this map silences libraries that spam at INFO/WARNING.
_NOISY_LOGGERS: dict[str, int] = {
    "sentence_transformers": logging.WARNING,
    "transformers": logging.WARNING,
    "datasets": logging.WARNING,
    "urllib3": logging.WARNING,
    # Nomic ships custom modeling code under a deeply-nested logger name and
    # prints "<All keys matched successfully>" as a WARNING. Suppress entirely.
    "transformers_modules": logging.ERROR,
}


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the root logger with a console handler and optional file handler.

    Safe to call multiple times: existing handlers are removed before
    re-installing fresh ones, so re-imports don't produce duplicate log lines.

    Parameters
    ----------
    log_file : str | Path | None
        If given, also append log records to this file. Parent directory is
        created if missing.
    level : int
        Logging level for both handlers. Defaults to INFO.
    """
    formatter = logging.Formatter(_FMT)

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name, lvl in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)

    return root


def relpath_str(p: str | Path, root: str | Path) -> str:
    """Return `p` as a forward-slash string relative to `root`.

    Falls back to `p.name` if `p` is not under `root`. Useful for logging
    paths without leaking absolute filesystem locations.
    """
    p = Path(p)
    root = Path(root)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.name
