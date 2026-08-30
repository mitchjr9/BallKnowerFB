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
    _configure_root_logger()
    return logging.getLogger(name or "ballknower_gridiron")
