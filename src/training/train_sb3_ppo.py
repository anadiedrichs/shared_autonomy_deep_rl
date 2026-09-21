"""
Entrena el baseline "framework tradicional": PPO vía Stable-Baselines3, sobre
el mismo entorno continuo que se usa para RLtools-SAC (src/training/
train_rltools_sac.py), para poder comparar de forma justa.

Análogo al código de la tesis 2025 (sección 3.2.3, código 3.1), adaptado de
acción discreta a continua. SB3 detecta automáticamente que el action_space es
Box y usa una política gaussiana, sin cambios adicionales de configuración.

TODO:
    - Portar la estructura del script de entrenamiento de la tesis 2025
      (instanciar entorno, PPO('MlpPolicy', env, ...), model.learn(...),
      model.save(...)).
    - Configurar el callback de Tensorboard (tensorboard_log=...), igual que en
      la tesis 2025.
"""

raise NotImplementedError("Pendiente de implementación, ver docstring del módulo.")
