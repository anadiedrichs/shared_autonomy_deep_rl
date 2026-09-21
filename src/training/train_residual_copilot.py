"""
Entrena la política residual del copiloto (src/copilot/residual_policy.py)
contra un piloto ya entrenado y congelado (óptimo, ruidoso o lento), usando
la recompensa goal-agnostic (src/rewards/reward_general_safety.py).

No hay equivalente directo en la tesis 2025.

TODO:
    - Cargar el piloto ya entrenado (OptimalPilot / NoisyPilot / LaggyPilot).
    - Instanciar SharedAutonomyEnv(pilot=..., copilot=<política en entrenamiento>).
    - Definir con qué backend se entrena el residual (RLtools-SAC o SB3, ver
      pregunta abierta en docs/architecture.md).
    - Definir si se entrena un único copiloto "generalista" contra varios tipos
      de piloto (ideal según Schaff & Walter, para no sobre-especializarse a un
      solo piloto) o uno por tipo de piloto, para la primera iteración.
"""

raise NotImplementedError("Pendiente de implementación, ver docstring del módulo.")
