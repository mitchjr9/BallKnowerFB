"""
ballknower_gridiron.utils.logging_utils
=======================================

Console + rotating file logging for the football package. Writes to
`logs/ballknower_gridiron.log` so it doesn't intermix with the tennis
or basketball log.
"""
from __future__ import annotations

import logging
import logging.handlers
from typing import Optional

from ballknower_gridiron.config.settings import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("ballknower_gridiron")
    root.setLevel(settings.log_level)
    root.propagate = False  # keep football logs out of tennis/hoops root logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(settings.log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(settings.log_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quiet noisy 3rd-party deps
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("nflreadpy").setLevel(logging.WARNING)
    logging.getLogger("polars").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger inside the `ballknower_gridiron` namespace.

    Handlers live only on the `ballknower_gridiron` logger. A logger created
    under any other name — `get_logger("weekly_football_pipeline")`, say — has
    no handler, so its INFO lines are silently discarded and only WARNING and
    above reach stderr via Python's last-resort handler. That is exactly how the
    pipeline's resolved-config banner went dark: it logs at INFO, so on every
    correctly configured run it printed nothing at all, and the one thing it
    existed to confirm was never shown. Prefixing bare names closes the gap for
    every caller at once instead of relying on each module to spell its name.
    """
    _configure_root_logger()
    root = "ballknower_gridiron"
    if not name:
        return logging.getLogger(root)
    if name != root and not name.startswith(root + "."):
        name = f"{root}.{name}"
    return logging.getLogger(name)
