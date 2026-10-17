# -*- coding: utf-8 -*-
"""
Mathematical and data processing utility functions for the drone and RL environments.
"""

import numpy as np


def normalize_to_range(
    value: float,
    min_val: float,
    max_val: float,
    new_min: float = -1.0,
    new_max: float = 1.0,
    clip: bool = False,
) -> float:
    """
    Normalizes a scalar numerical value from a source range [min_val, max_val]
    to a target range [new_min, new_max].

    Args:
        value: Numerical value to normalize.
        min_val: Lower bound of the source range.
        max_val: Upper bound of the source range.
        new_min: Lower bound of the target range (default -1.0).
        new_max: Upper bound of the target range (default 1.0).
        clip: If True, clips the result within [new_min, new_max].

    Returns:
        float: Normalized value in the target range.
    """
    val = float(value)
    min_v = float(min_val)
    max_v = float(max_val)
    new_min_v = float(new_min)
    new_max_v = float(new_max)

    if max_v == min_v:
        return new_min_v

    normalized = ((new_max_v - new_min_v) / (max_v - min_v)) * (val - max_v) + new_max_v

    if clip:
        return float(np.clip(normalized, new_min_v, new_max_v))
    return float(normalized)
