"""Compatibility wrapper for importing the src-layout package from repo root."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "nba_championship_prediction"

if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))

from .cli import main
from .modeling import AttentionModel, MLPBaseline, PLAYOFF_CLASS_NAMES, PLAYER_SLOT_NAMES

__all__ = [
    "AttentionModel",
    "MLPBaseline",
    "PLAYOFF_CLASS_NAMES",
    "PLAYER_SLOT_NAMES",
    "main",
]
