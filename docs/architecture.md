# Decisiones de arquitectura

Este documento registra las decisiones tomadas antes de portar código desde
[`drone-deep-rl`](https://github.com/anadiedrichs/drone-deep-rl) (tesis CEIA-FIUBA 2025) a
este repositorio, y el razonamiento detrás de cada una. El objetivo es que cualquier
colaborador (o la propia autora, meses después) entienda *por qué* se descartaron otras
opciones, no solo cuál se eligió.

## 1. Integración RLtools–Webots: wrapper Python (no C++ nativo)

RLtools es una librería **header-only en C++ puro**, sin dependencias, diseñada para que
el compilador conozca en tiempo de compilación las dimensiones de todas las estructuras
(observaciones, acciones, capas de la red). El equipo de RLtools distribuye también un
wrapper Python (`pip install rltools`, repo
[`rl-tools/python-interface`](https://github.com/rl-tools/python-interface)) que:

- Depende solo de `pybind11`.
- Toma como entrada una función `env_factory()` que devuelve un entorno **Gymnasium**, e
  infiere de ahí las dimensiones de observación/acción para compilar en tiempo de
  ejecución un entorno "puente" compatible con RLtools.
- Hoy **solo expone el loop de entrenamiento de SAC**, sin mucho control sobre
  hiperparámetros ("work in progress").
- Permite exportar el checkpoint entrenado como **header C** (`state.export_policy()`),
  pensado para desplegarse directo en firmware embebido — la pieza que conecta con el
  objetivo de portabilidad / sim-to-real del proyecto, incluso sin escribir el
  controlador de Webots en C++.

### Opciones consideradas

| Opción | Descripción | Por qué no se eligió (o se eligió) |
|---|---|---|
| A | Controlador Webots en C++ puro + RLtools nativo | Máximo rendimiento y fidelidad al objetivo TinyRL, pero implica reescribir todo `DroneRobotSupervisor`, sensores, PID y pilotos en C++ desde cero. Mayor riesgo de cronograma. Queda como evolución futura una vez validada la arquitectura en Python. |
| B | Wrapper Python de RLtools | **Elegida.** Reusa el `DroneRobotSupervisor` en Python casi tal cual. Limitación conocida: solo SAC por ahora. |
| C | Híbrido (SB3 en Python + módulo C++ aparte con RLtools) | Descartada por duplicar mantenimiento; se prefirió una única base de código en Python (opción B) más SB3 como baseline dentro del mismo stack. |

### Consecuencia sobre el objetivo "comparar PPO y SAC integrados nativamente con RLtools"

El wrapper Python de RLtools no entrena con PPO. Por lo tanto, la comparación PPO vs. SAC
se resuelve así:

- **SAC vía `rltools`**: candidato a "modelo optimizado/portable" (política que se puede
  exportar a C y potencialmente desplegar en el Crazyflie real).
- **PPO vía `stable-baselines3`**: baseline de "framework tradicional", que además es
  exactamente lo que pide el objetivo de evaluación del proyecto ("comparar... frente a
  pilotos de línea base y modelos entrenados en frameworks tradicionales").

Esto se documenta explícitamente para que en la memoria/paper final quede claro que la
asimetría (SAC-RLtools vs. PPO-SB3, en lugar de PPO/SAC ambos en RLtools) es una
limitación conocida de la librería en su versión actual, no un error de diseño.

## 2. Piloto óptimo: RL puro, sin aprendizaje por imitación (por ahora)

En la tesis 2025, el piloto óptimo se obtuvo con dos enfoques: RL (PPO/DQN/A2C con
acciones discretas) y aprendizaje por imitación (Behavioral Cloning con la librería
`imitation`, a partir de datos capturados con teclado).

Al pasar a acciones **continuas**, la captura de demostraciones con teclado deja de ser
directa: sin joystick, solo se pueden capturar comandos "bang-bang" (valor máximo
mientras se mantiene la tecla, cero al soltar), que no explotan el rango continuo del
espacio de acciones.

### Opciones consideradas para el piloto óptimo

1. **RL puro (elegida)**: RLtools-SAC + SB3-PPO, sin captura humana. Es coherente con el
   objetivo específico principal del proyecto (entrenar y comparar PPO/SAC), y no
   depende de conseguir hardware.
2. Captura "bang-bang" con teclado, mapeada a los extremos de `Box(-1,1)`: descartada por
   ahora, queda como posible fallback de BC si RL no converge bien en algún escenario
   (igual que ocurrió en la tesis 2025 con PPO puro).
3. Joystick virtual (mouse/trackpad → ejes continuos): es una tarea **opcional** listada
   en el formulario del proyecto ("Desarrollar la infraestructura de control..."). No se
   implementa en esta primera iteración.
4. Gamepad/joystick USB físico: requiere adquisición de hardware, fuera del alcance
   inmediato.

## 3. Espacio de acciones continuo

```
Box(low=-1, high=1, shape=(4,), dtype=float32)
# [vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd] (normalizado)
```

Se reutiliza el controlador PID de velocidad ya implementado en `DroneRobotSupervisor`
(tesis 2025): las 6 acciones discretas originales eran combinaciones fijas de estos 4
valores, así que el paso a continuo no requiere rediseñar el controlador de bajo nivel,
solo desnormalizar el vector de acción antes de pasarlo al PID (mismo patrón que usa el
propio ejemplo de RLtools con `gymnasium.wrappers.RescaleAction`).

## 4. Autonomía compartida: política residual

Siguiendo [Schaff & Walter (2020)](https://arxiv.org/abs/2004.05097):

```
a = a_h + a_r
a_h ~ pi_h(s, g)      # piloto: humano o sintético (óptimo / ruidoso / lento)
a_r ~ pi_r(s, a_h)    # copiloto: corrección residual, condicionada a estado y acción del piloto
```

Esto reemplaza el esquema discreto de la tesis 2025 (tabla de distancia `Q(a*, ah)` +
umbral `alpha`, que elegía *entre* la acción del piloto y la del copiloto) por una
combinación *aditiva* y continua, donde el copiloto:

- Se entrena contra una recompensa **agnóstica a la meta** (`reward_general_safety.py`):
  penaliza colisión/proximidad a obstáculos y penaliza la magnitud `‖a_r‖²`, para que
  intervenga lo mínimo posible.
- No conoce la meta del piloto (a diferencia del piloto, que sí se entrena con una
  recompensa orientada a la meta, `reward_goal_directed.py`).

## 5. Modelo de ejecución Webots: Controlador Externo (`<extern>`)

En la tesis 2025 (`drone-deep-rl`), el flujo de trabajo probado consistía en abrir Webots y
**ejecutar los scripts de control y entrenamiento directamente desde el IDE (Visual Studio Code)**
o la terminal.

Para mantener esta operatividad y desacoplar el entorno Gymnasium de la interfaz gráfica de Webots,
el nodo Crazyflie en el mundo `.wbt` (`webots_project/worlds/crazyflie.wbt`) se configura con:

```vrml
DEF crazyflie Crazyflie {
  ...
  controller "<extern>"
  supervisor TRUE
}
```

### Justificación técnica:

1. **Inversión de control requerida por Deep RL**: En frameworks como Stable-Baselines3 (PPO) y
   RLtools (SAC), el bucle principal de control pertenece al algoritmo de aprendizaje
   (`model.learn()` y `env.step()`), no al simulador. Al operar como controlador externo, el proceso
   Python actúa como proceso maestro que avanza la simulación física paso a paso vía IPC
   (`supervisor.step(timestep)`), permitiendo pausar el simulador durante el cálculo de gradientes
   y optimizaciones en GPU.
2. **Flexibilidad de ejecución (IDE / Terminal / Docker)**: Permite ejecutar scripts de evaluación
   ([`src/evaluation/run_heuristic_episode.py`](file:///home/ana/0001-webots-rl/src/evaluation/run_heuristic_episode.py)) o entrenamiento
   desde cualquier terminal, VS Code o contenedor Docker (`webots_rltools_env`), conectándose
   automáticamente a la simulación física activa por memoria compartida (`ipc: host`).
3. **Simulación acelerada sin renderizado (Headless)**: Hace posible entrenar a velocidades muy
   superiores al tiempo real (10x-50x) mediante la opción headless de Webots:
   `webots --batch --mode=fast --no-rendering <mundo.wbt>`.

## Preguntas abiertas / pendientes de confirmar

- Parametrización exacta de `reward_general_safety.py`: pesos relativos entre penalización
  de colisión y penalización de `‖a_r‖²`.
- Si el residual se entrena con RLtools-SAC (coherente con el resto del pipeline) o con
  SB3, y si se reentrena por cada tipo de piloto (óptimo/ruidoso/lento) o con un piloto
  fijo durante el entrenamiento y se evalúa contra los demás.
- Cómo exportar/visualizar las curvas de entrenamiento de RLtools en Tensorboard (SB3 ya
  lo soporta nativamente; RLtools tiene su propio logging a revisar).
