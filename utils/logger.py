"""
utils/logger.py
================
Structured logging setup using structlog.

Provides consistent, JSON-structured log output that is:
  - Machine-readable (for log aggregation tools)
  - Human-readable in console (with rich formatting)
  - Configurable via logging_config.yaml

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened", key=value, another=42)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


_configured = False


def configure_logging(
    level: str = "INFO",
    log_format: str = "console",
    log_file: Optional[str] = None,
    rich_formatting: bool = True,
) -> None:
    """
    Configure the global logging setup.

    Must be called once at application startup (in main.py).

    Parameters
    ----------
    level : str
        Log level: "DEBUG", "INFO", "WARNING", "ERROR".
    log_format : str
        "console" for human-readable, "json" for machine-readable.
    log_file : Optional[str]
        Path to write logs to disk. None = console only.
    rich_formatting : bool
        Use rich colours and formatting in console output.
    """
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger setup
    handlers: list = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
    )

    # Structlog processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        if rich_formatting:
            renderer = structlog.dev.ConsoleRenderer(colors=True)
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a named logger instance.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.

    Returns
    -------
    BoundLogger
        Structlog bound logger with the module name pre-bound.
    """
    if not _configured:
        # Auto-configure with defaults if configure_logging() wasn't called
        configure_logging(level="INFO", log_format="console")
    return structlog.get_logger(name)


def set_context(**kwargs: object) -> None:
    """
    Bind key-value pairs to the current async context (e.g. episode_date, agent_run_id).
    All subsequent log calls in this context will include these values.

    Parameters
    ----------
    **kwargs
        Context variables to bind.

    Example
    -------
    set_context(episode_date="2023-03-15", agent_run_id="run_42")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()