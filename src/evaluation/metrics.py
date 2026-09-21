"""
Métricas de evaluación, análogas a las tablas 4.1-4.4 de la tesis 2025:
    - Recompensa promedio y desviación estándar por episodio.
    - Longitud promedio del episodio y desviación estándar.
    - Tasa de éxito (llegó al objetivo).
    - Tasa de choques.

Estas métricas ya eran independientes de si la acción era discreta o continua
en la tesis 2025 (se calculaban sobre resultado del episodio, no sobre la
acción en sí), así que este es mayormente un port directo de la lógica de
evaluación (no de una clase puntual, sino del patrón usado en las secciones
4.2-4.4 de la memoria).

TODO:
    - Portar la función que corre N episodios con una política/piloto dado y
      agrega estas métricas (equivalente a lo usado para generar las tablas
      4.1-4.4 de la tesis 2025).
"""

import numpy as np


def evaluate_pilot(env, pilot, n_episodes: int = 100, seed: int | None = None) -> dict:
    """
    TODO: correr n_episodes con `pilot` sobre `env`, registrar por episodio:
    recompensa acumulada, longitud, éxito (bool), choque (bool). Devolver un
    dict con promedio/desviación estándar de cada métrica, como en las tablas
    4.1-4.4 de la tesis 2025.
    """
    raise NotImplementedError
