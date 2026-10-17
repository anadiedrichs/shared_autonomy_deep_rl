# -*- coding: utf-8 -*-
"""
Module of synthetic and learned pilots for the Crazyflie.
"""

from src.pilots.base_pilot import BasePilot, Pilot
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot

__all__ = [
    "BasePilot",
    "Pilot",
    "OptimalPilot",
    "NoisyPilot",
    "LaggyPilot",
    "HeuristicPilot",
]
