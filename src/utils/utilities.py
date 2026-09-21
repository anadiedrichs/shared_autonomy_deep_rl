# -*- coding: utf-8 -*-
"""
Funciones de utilidad matemática y procesamiento para el dron y entornos de RL.
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
    Normaliza un valor numérico escalar desde un rango de origen [min_val, max_val]
    hacia un nuevo rango objetivo [new_min, new_max].

    Args:
        value: Valor numérico a normalizar.
        min_val: Límite inferior del rango de origen.
        max_val: Límite superior del rango de origen.
        new_min: Límite inferior del rango destino (por defecto -1.0).
        new_max: Límite superior del rango destino (por defecto 1.0).
        clip: Si es True, recorta el resultado dentro de [new_min, new_max].

    Returns:
        float: Valor normalizado en el rango destino.
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
