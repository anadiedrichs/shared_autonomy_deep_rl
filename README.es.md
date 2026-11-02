*Leer en otros idiomas: [English](README.md)*

# Shared Autonomy Deep RL

Framework de integración entre [RLtools](https://github.com/rl-tools/rl-tools) y el simulador
[Webots](https://cyberbotics.com/) para implementar un sistema de teleoperación asistida
(autonomía compartida) de cuadricópteros Crazyflie, mediante aprendizaje de políticas
residuales en espacio de acciones continuo.

Continuación del trabajo de tesis de Ana Laura Diedrichs, *"Teleoperación asistida de
cuadricópteros mediante aprendizaje por refuerzo profundo"* (CEIA-FIUBA, 2025,
[Zenodo](https://doi.org/10.5281/zenodo.16914761)), y del repositorio
[drone-deep-rl](https://github.com/anadiedrichs/drone-deep-rl), en el marco del proyecto
de investigación *"Autonomía Compartida para UAVs: desarrollo Sim-to-Real con Webots y
RLtools"* (Universidad Europea).

> **Estado del Proyecto**: **Framework Completo de Trabajo de Fin de Máster (TFM)**. Políticas continuas de navegación completamente implementadas (RLtools TD3/SAC y SB3 PPO), modelos sintéticos de pilotos humanos degradados (NoisyPilot, LaggyPilot, StandByPilot), y Copiloto Residual Constreñido con multiplicadores adaptativos de Lagrange (Schaff & Walter, 2020). Los modelos experimentales, evaluaciones y curvas de telemetría se encuentran consolidados en `results_TFM/`.

---

## 🔬 Guía Rápida de Reproducibilidad (Para Evaluadores)

Para reproducir de forma inmediata todas las tablas estadísticas (Tablas 4.1 a 4.5) y figuras de la memoria de TFM sin necesidad de instalar Webots ni ejecutar entrenamientos pesados:

```bash
# Nivel 1: Regeneración offline inmediata (< 1 min, sin dependencias externas pesadas)
python3 scripts/reproduce_results.py
```

O explorando el cuaderno interactivo de Jupyter:
* [`notebooks/reproduce_thesis_results.ipynb`](notebooks/reproduce_thesis_results.ipynb)

Para consultar el protocolo completo de reproducibilidad en 3 niveles (incluyendo evaluación en Webots y entrenamiento desde cero), revisa el documento maestro:
👉 [**REPRODUCIBILITY.md**](REPRODUCIBILITY.md) (o [Versión en Español](REPRODUCIBILIDAD.md)).

---

## Decisiones de Arquitectura (Resumen)

- **Stack**: Python puro con aceleración C++. Webots + Gymnasium, aprovechando el motor C++ de alto rendimiento de [RLtools](https://github.com/rl-tools/rl-tools) para despliegue estático libre de dependencias junto con PyTorch / Stable-Baselines3.
- **Piloto óptimo**: 100% aprendizaje por refuerzo continuo. Evaluado sobre tres algoritmos actor-crítico:
  - [`rltools`](https://pypi.org/project/rltools/) → TD3 y SAC (formato estático `.h` para embebidos).
  - [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) → PPO baseline (`.zip`).
- **Espacio de acciones**: continuo, `Box(-1, 1, shape=(4,))` =
  `[vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd]`, reutilizando el controlador PID de
  velocidad validado en `drone-deep-rl`.
- **Copiloto (autonomía compartida)**: Política Residual Constreñida, siguiendo
  [Schaff & Walter, 2020](https://arxiv.org/abs/2004.05097):

  ```
  a = clip(a_h + beta * a_r, -1, 1)
  a_h ~ pi_h(s, g)      # piloto (humano o sintético): orientado a la meta
  a_r ~ pi_r(s, a_h)    # copiloto: corrección residual, agnóstica a la meta
  ```

  El copiloto minimiza su propia intervención (`‖a_r‖²`) mientras garantiza restricciones críticas de seguridad (proximidad a obstáculos, paredes perimetrales y prevención de caídas) reguladas dinámicamente mediante un multiplicador de Lagrange.

Ver la discusión detallada de arquitectura en [`docs/architecture.md`](docs/architecture.md).

---

## Estructura del Proyecto

```
0001-webots-rl/
├── REPRODUCIBILITY.md         # Guía maestra de reproducibilidad de resultados (3 niveles)
├── REPRODUCIBILIDAD.md        # Enlace a la guía de reproducibilidad
├── webots_project/            # Mundo (.wbt) y controlador supervisor Webots
├── results_TFM/               # Modelos experimentales (.h, .zip) y evaluaciones CSV
│   ├── models/                # Checkpoints entrenados (TD3, SAC, PPO, Copiloto)
│   └── logs/                  # Logs multisemilla de 100 episodios y trayectorias
├── scripts/                   # Scripts automatizados de entrenamiento y evaluación
│   ├── reproduce_results.py   # Script autónomo de reproducción de tablas y figuras
│   ├── train_all_seeds.sh     # Batería de entrenamiento multisemilla
│   ├── evaluate_degraded_pilots.sh # Evaluación multisemilla de pilotos degradados
│   ├── train_copilot.sh       # Entrenamiento del copiloto PPO constreñido
│   └── evaluate_copilot.sh    # Evaluación comparativa Placebo vs. Asistido
├── notebooks/                 # Cuadernos interactivos Jupyter para análisis
│   ├── reproduce_thesis_results.ipynb  # Reproducción interactiva de resultados
│   └── 01_pilots_and_copilot_analysis.ipynb
├── src/
│   ├── envs/                  # Entornos Gymnasium continuos y SharedAutonomyEnv
│   ├── pilots/                # BasePilot, OptimalPilot, NoisyPilot, LaggyPilot, StandByPilot
│   ├── copilot/               # ConstrainedPPO (política residual con Lagrange)
│   ├── rewards/               # Funciones de recompensa orientadas a meta y seguridad
│   ├── training/              # Flujos de entrenamiento (TD3, SAC, SB3 PPO, Copiloto)
│   └── evaluation/            # Métricas cuantitativas, graficado de trayectorias
├── docs/                      # Guías de arquitectura, Docker, entrenamiento y evaluación
└── INSTALL.md                 # Instrucciones de instalación y despliegue
```

---

## Instalación y Ejecución en Simulador

Consulta [`INSTALL.md`](INSTALL.md) y [`docs/docker_guide.md`](docs/docker_guide.md).

---

## Referencias Técnicas

Diedrichs, Ana Laura. «Teleoperación asistida de cuadricópteros mediante aprendizaje por
refuerzo profundo». Zenodo, 2025. <https://doi.org/10.5281/zenodo.16914761>

Schaff, C., & Walter, M. R. (2020). Residual Policy Learning for Shared Autonomy.
*Robotics: Science and Systems (RSS)*. <https://arxiv.org/abs/2004.05097>

Eschmann, J., Albani, D., & Loianno, G. (2024). RLtools: A Fast, Portable Deep
Reinforcement Learning Library for Continuous Control. *JMLR*, 25(301).
<http://jmlr.org/papers/v25/24-0248.html>

---

## Licencia

Apache-2.0. Ver [`LICENSE`](LICENSE).

## Contacto

Ana Diedrichs - ana (dot) diedrichs (at) docentes.frm.utn.edu.ar
