"""Fixed per-problem objective scales for preference aggregation (JCDE revision, preference model v2).

Every achievement scalarizing function (host and simulated decision maker) divides the weighted objective
deviations by these scales, so that objectives with different units carry comparable weight. The default
(None) keeps the original unscaled behaviour bit for bit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_SCALE: Optional[np.ndarray] = None


def set_scale(scale) -> None:
    global _SCALE
    _SCALE = None if scale is None else np.asarray(scale, dtype=float).ravel()


def get_scale() -> Optional[np.ndarray]:
    return _SCALE


def scaled(d: np.ndarray) -> np.ndarray:
    """Divide per-objective deviations by the active scales (identity when no scale is set)."""
    if _SCALE is None:
        return d
    return d / _SCALE[: len(d)]
