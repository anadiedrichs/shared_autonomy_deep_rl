"""
Tests del entorno continuo. Por ahora solo valida la forma del action_space;
se irán agregando a medida que se implemente cada módulo (ver TODOs en
src/envs/).

TODO:
    - Test de action_space.shape == (4,) y rango [-1, 1].
    - Test de reset()/step() contra gymnasium.utils.env_checker.check_env
      (hito de fin de Fase 1 según docs/architecture.md).
"""

import pytest


def test_placeholder():
    """Placeholder: reemplazar cuando DroneRobotSupervisor esté implementado."""
    pytest.skip("Pendiente: DroneRobotSupervisor aún no implementado.")
