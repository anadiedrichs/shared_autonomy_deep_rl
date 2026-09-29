"""
Entorno que compone piloto + copiloto residual (autonomía compartida).

No existe un equivalente directo en la tesis 2025 (ahí la composición vivía
dentro de get_action() de DroneRobotSupervisor, con un esquema de selección
discreta piloto-o-copiloto vía la función de calidad Q). Este módulo es nuevo,
ver docs/architecture.md sección 4 para la justificación del cambio de diseño.

Esquema (Schaff & Walter, 2020):

    a = clip(a_h + a_r, -1, 1)
    a_h ~ pi_h(s, g)      # piloto: humano o sintético (óptimo / ruidoso / lento)
    a_r ~ pi_r(s, a_h)    # copiloto: corrección residual

TODO:
    - Decidir la interfaz exacta del "pilot" inyectado (¿objeto con método
      get_action(obs) -> np.ndarray, como en la tesis 2025?).
    - Decidir si el copiloto (residual_policy) se pasa ya entrenado (modo
      evaluación) o se entrena dentro de este mismo wrapper (modo entrenamiento,
      con la observación extendida para incluir a_h como en la tesis 2025).
"""

import numpy as np

from src.envs.corner_env_continuous import CornerEnvContinuous


class SharedAutonomyEnv(CornerEnvContinuous):
    """
    Envuelve CornerEnvContinuous agregando la composición piloto + copiloto.
    """

    def __init__(self, pilot, copilot=None, max_episode_steps: int = 1_000):
        """
        Args:
            pilot: objeto con método get_action(observation) -> np.ndarray(4,),
                en [-1, 1]. P.ej. OptimalPilot, NoisyPilot, LaggyPilot.
            copilot: objeto con método get_action(observation, pilot_action) ->
                np.ndarray(4,) (el residual a_r). None = sin asistencia (solo
                piloto).
        """
        super().__init__(max_episode_steps=max_episode_steps)
        self.pilot = pilot
        self.copilot = copilot

    def step(self, action=None):
        """
        TODO:
            1. obs = observación actual
            2. a_h = self.pilot.get_action(obs)
            3. a_r = self.copilot.get_action(obs, a_h) if self.copilot else 0
            4. a = np.clip(a_h + a_r, -1, 1)
            5. return super().step(a)
        """
        raise NotImplementedError
