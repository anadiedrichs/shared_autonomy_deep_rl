# -*- coding: utf-8 -*-
"""
Módulo de pilotos sintéticos y aprendidos para Crazyflie.
"""

from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot

__all__ = [
    "OptimalPilot",
    "NoisyPilot",
    "LaggyPilot",
    "HeuristicPilot",
]
