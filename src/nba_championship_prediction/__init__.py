"""NBA championship prediction package."""

from .cli import main
from .modeling import AttentionModel, MLPBaseline, PLAYOFF_CLASS_NAMES, PLAYER_SLOT_NAMES

__all__ = [
    "AttentionModel",
    "MLPBaseline",
    "PLAYOFF_CLASS_NAMES",
    "PLAYER_SLOT_NAMES",
    "main",
]
