# -*- coding: utf-8 -*-
"""
Tests unitarios para la integración de RLtools SAC y exportación de C Header.
"""

import os
import sys
import tempfile
import numpy as np

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.training.train_rltools_sac import make_env_factory
from src.pilots.optimal_pilot import OptimalPilot


def test_make_env_factory_singleton():
    """Valida que la fábrica de entorno devuelva una instancia persistente (singleton)."""
    factory = make_env_factory(corner_name="cone_1", max_episode_steps=500)
    env1 = factory()
    env2 = factory()

    assert env1 is env2, "La fábrica debe devolver la misma instancia singleton para evitar múltiples Supervisores"
    assert env1.action_space.shape == (4,)
    assert env1.observation_space.shape == (11,)


def test_rltools_sac_workflow_mock():
    """Valida el ciclo de entrenamiento e inferencia de RLtools SAC con exportación a C header."""
    try:
        import rltools
    except ImportError:
        print("[Test] rltools no disponible en este entorno, saltando test_rltools_sac_workflow_mock.")
        return

    factory = make_env_factory(corner_name="cone_1", max_episode_steps=500)

    # 1. Instanciar SAC con límite pequeño para test unitario rápido
    sac = rltools.SAC(
        factory,
        STEP_LIMIT=1000,
        EPISODE_STEP_LIMIT=500,
        GAMMA=0.97,
        evaluation_interval=1000,
        num_evaluation_episodes=2,
        verbose=False,
    )
    state = sac.State(42)

    # 2. Ejecutar un paso de entrenamiento
    finished = state.step()
    assert isinstance(finished, bool)

    # 3. Validar exportación de política C
    c_header = state.export_policy()
    assert isinstance(c_header, str)
    assert len(c_header) > 100
    assert "namespace" in c_header or "struct" in c_header or "rl_tools" in c_header

    # 4. Validar carga e inferencia mediante OptimalPilot
    with tempfile.NamedTemporaryFile(suffix=".h", mode="w", delete=False) as f:
        f.write(c_header)
        temp_header_path = f.name

    try:
        pilot = OptimalPilot(backend="rltools_sac", checkpoint_path=temp_header_path)
        obs = np.zeros(11, dtype=np.float32)
        act = pilot.get_action(obs)

        assert isinstance(act, np.ndarray)
        assert act.shape == (4,)
        assert np.all(act >= -1.0) and np.all(act <= 1.0)
    finally:
        if os.path.exists(temp_header_path):
            os.remove(temp_header_path)


if __name__ == "__main__":
    test_make_env_factory_singleton()
    test_rltools_sac_workflow_mock()
    print("Todos los tests de RLtools SAC pasaron exitosamente.")
