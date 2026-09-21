"""
Recompensa goal-agnostic (R_general), para entrenar al copiloto (política
residual), siguiendo Schaff & Walter (2020): el copiloto se entrena para
minimizar su propia intervención mientras se satisfacen restricciones de
seguridad, sin conocer la meta del piloto.

No hay equivalente directo en la tesis 2025 (la función de recompensa original
era toda orientada a la meta). Este módulo es nuevo.

Componentes propuestos (a confirmar, ver pregunta abierta en
docs/architecture.md):
    1. Penalización fuerte por colisión / proximidad a obstáculos (reusar
       walls_proximity_penalization de reward_goal_directed.py).
    2. Penalización por magnitud del residual: - lambda * ||a_r||^2, para que
       el copiloto intervenga lo mínimo posible.

TODO:
    - Definir el valor de lambda (peso relativo entre seguridad e intervención
      mínima).
    - Decidir si esta recompensa también incluye el término de éxito
      (llegar a la meta) o es estrictamente agnóstica a la meta, como en el
      paper original.
"""


def general_safety_reward(distance_to_obstacle: float, residual_action, lambda_intervention: float = 0.1) -> float:
    """
    TODO:
        penalty_collision = walls_proximity_penalization(distance_to_obstacle)
        penalty_intervention = lambda_intervention * np.sum(residual_action ** 2)
        return penalty_collision - penalty_intervention
    """
    raise NotImplementedError
