"""
Centralized logging setup. Logs to console + rotating file in logs/.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config import settings

_LOGGERS = {}


def get_logger(name: str = "rag") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    os.makedirs(settings.misc.log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.misc.log_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            os.path.join(settings.misc.log_dir, "app.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
