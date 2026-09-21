"""
Entry point del controlador Webots. Webots ejecuta este archivo directamente
(configurado como controlador del robot Crazyflie en el mundo .wbt).

TODO:
    - Importar SharedAutonomyEnv y el piloto/copiloto correspondientes.
    - Instanciar el entorno, cargar checkpoints entrenados.
    - Loop principal: obs = env.reset(); while True: obs, r, done, info =
      env.step(); if done: obs = env.reset().
    - Asegurarse de que el Python command configurado en Webots (Tools ->
      Preferences -> General -> Python command) apunte al venv del proyecto
      (ver INSTALL.md).
"""

# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# from src.envs.shared_autonomy_env import SharedAutonomyEnv
# from src.pilots.optimal_pilot import OptimalPilot

raise NotImplementedError("Pendiente de implementación.")
