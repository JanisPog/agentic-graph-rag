"""Structured PIVOT decision logger — one log line per stage tagged [PIVOT]."""
from __future__ import annotations
import logging

log = logging.getLogger("pivot")


def pivot_log(stage: str, decision: str, reason: str = "", **fields) -> None:
    extras = "  ".join(f"{k}={v}" for k, v in fields.items())
    log.info("[PIVOT] %-8s decision=%-18s reason=%-32s %s",
             stage.upper(), decision, reason or "-", extras)
