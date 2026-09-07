"""Lightweight physics-guided TESS TCE classifier."""

from .models.pgnn import LightweightPGNN, PGNNConfig

__all__ = ["LightweightPGNN", "PGNNConfig"]
__version__ = "0.1.0"
