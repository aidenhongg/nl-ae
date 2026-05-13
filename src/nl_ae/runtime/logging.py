"""Stdlib logging configuration: stderr console + JSONL file handler."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

_JSONL_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


class JsonlFormatter(logging.Formatter):
    """One JSON object per line. ``record.extra`` keys are merged at the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _JSONL_RESERVED:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    *,
    run_log_dir: Path | None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    per_module_levels: dict[str, str] | None = None,
    jsonl_destination: Literal["run_dir", "stderr", "off"] = "run_dir",
) -> logging.Logger:
    """Configure the root logger; return the ``nl_ae`` logger."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Idempotent: drop any handlers we previously installed.
    for h in list(root.handlers):
        if getattr(h, "_nlae_handler", False):
            root.removeHandler(h)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, console_level, logging.INFO))
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)s — %(message)s", "%H:%M:%S")
    )
    setattr(console, "_nlae_handler", True)
    root.addHandler(console)

    if jsonl_destination == "run_dir":
        if run_log_dir is None:
            raise ValueError("run_log_dir is required when jsonl_destination='run_dir'")
        run_log_dir.mkdir(parents=True, exist_ok=True)
        file_handler: logging.Handler = logging.FileHandler(
            run_log_dir / "run.log.jsonl", encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, file_level, logging.DEBUG))
        file_handler.setFormatter(JsonlFormatter())
        setattr(file_handler, "_nlae_handler", True)
        root.addHandler(file_handler)
    elif jsonl_destination == "stderr":
        json_handler = logging.StreamHandler(sys.stderr)
        json_handler.setLevel(getattr(logging, file_level, logging.DEBUG))
        json_handler.setFormatter(JsonlFormatter())
        setattr(json_handler, "_nlae_handler", True)
        root.addHandler(json_handler)
    # "off" => only the console handler.

    for name, level in (per_module_levels or {}).items():
        logging.getLogger(name).setLevel(getattr(logging, level, logging.INFO))

    return logging.getLogger("nl_ae")


__all__ = ["JsonlFormatter", "setup_logging"]
