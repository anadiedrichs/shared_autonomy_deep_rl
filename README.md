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

> Estado: **esqueleto inicial**. Los módulos en `src/` contienen la interfaz y los
> docstrings de diseño, pendientes de implementación. Ver `docs/architecture.md` para el
> detalle de las decisiones de arquitectura tomadas antes de portar el código.

## Decisiones de arquitectura (resumen)

- **Stack**: Python puro. Webots + Gymnasium, sin controlador C++ nativo por ahora.
- **Piloto óptimo**: 100% aprendizaje por refuerzo. Se entrena con dos algoritmos para
  poder comparar:
  - [`rltools`](https://pypi.org/project/rltools/) (wrapper Python de RLtools) → SAC.
  - [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) → PPO, como
    baseline "framework tradicional".
  No se usa aprendizaje por imitación en esta iteración (no hay joystick disponible para
  capturar demostraciones continuas de buena calidad; queda como trabajo futuro).
- **Espacio de acciones**: continuo, `Box(-1, 1, shape=(4,))` =
  `[vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd]`, reutilizando el controlador PID de
  velocidad ya validado en `drone-deep-rl`.
- **Copiloto (autonomía compartida)**: política residual, siguiendo
  [Schaff & Walter, 2020](https://arxiv.org/abs/2004.05097):

  ```
  a = clip(a_h + a_r, -1, 1)
  a_h ~ pi_h(s, g)      # piloto (humano o sintético): orientado a la meta
  a_r ~ pi_r(s, a_h)    # copiloto: corrección residual, agnóstica a la meta
  ```

  El copiloto se entrena para minimizar su propia intervención (`‖a_r‖²`) mientras
  evita violar restricciones de seguridad (colisiones / proximidad a obstáculos).

Ver la discusión completa de por qué se eligió esta arquitectura (y no RLtools nativo en
C++, ni aprendizaje por imitación) en `docs/architecture.md`.

## Estructura del proyecto

```
shared_autonomy_deep_rl/
├── webots_project/            # Mundo (.wbt) y controlador Webots
├── src/
│   ├── envs/                  # Entornos Gymnasium (drone, escenario, composición piloto+copiloto)
│   ├── pilots/                # Piloto óptimo, ruidoso, lento (sintéticos)
│   ├── copilot/                # Política residual
│   ├── rewards/                # Funciones de recompensa (orientada a meta / goal-agnostic)
│   ├── training/                # Scripts de entrenamiento (RLtools-SAC, SB3-PPO, copiloto)
│   └── evaluation/              # Métricas y comparación de pilotos
├── notebooks/                  # Análisis exploratorio de resultados
├── tests/
├── docs/architecture.md        # Decisiones de diseño y su justificación
├── requirements.txt
└── INSTALL.md
```

## Instalación

Ver [`INSTALL.md`](INSTALL.md).

## Documento técnico de referencia

Diedrichs, Ana Laura. «Teleoperación asistida de cuadricópteros mediante aprendizaje por
refuerzo profundo». Zenodo, 2025. <https://doi.org/10.5281/zenodo.16914761>

Schaff, C., & Walter, M. R. (2020). Residual Policy Learning for Shared Autonomy.
*Robotics: Science and Systems (RSS)*. <https://arxiv.org/abs/2004.05097>

Eschmann, J., Albani, D., & Loianno, G. (2024). RLtools: A Fast, Portable Deep
Reinforcement Learning Library for Continuous Control. *JMLR*, 25(301).
<http://jmlr.org/papers/v25/24-0248.html>

## Licencia

Apache-2.0. Ver [`LICENSE`](LICENSE).

## Contacto

Ana Diedrichs - ana (dot) diedrichs (at) docentes.frm.utn.edu.ar
