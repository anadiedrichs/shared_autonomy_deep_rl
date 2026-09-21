"""
Entrena el piloto óptimo con RLtools, algoritmo SAC.

Patrón de uso de la librería (confirmado contra el README oficial de
rl-tools/python-interface):

    from rltools import SAC
    import gymnasium as gym
    from gymnasium.wrappers import RescaleAction

    def env_factory():
        env = CornerEnvContinuous()          # nuestro entorno
        env = RescaleAction(env, -1, 1)      # asegura rango [-1, 1] exacto
        return env

    sac = SAC(env_factory)
    state = sac.State(seed)

    finished = False
    while not finished:
        finished = state.step()

    # Exportar checkpoint como header C (para posible despliegue embebido)
    with open("optimal_pilot_sac_checkpoint.h", "w") as f:
        f.write(state.export_policy())

TODO:
    - Reemplazar el env_factory de ejemplo por CornerEnvContinuous real.
    - Definir la semilla y el criterio de "finished" (número de pasos /
      episodios, o convergencia).
    - Revisar cómo loggear métricas de este loop a Tensorboard (RLtools tiene
      su propio logging interno; ver pregunta abierta en docs/architecture.md).
"""

raise NotImplementedError("Pendiente de implementación, ver docstring del módulo.")
