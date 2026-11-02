# Training guide

El how-to para entrenar y visualizar el entrenamiento en Webots.

Aprenderás a entrenar y evaluar el piloto óptimo continuo utilizando **PPO (Stable-Baselines3)**, **SAC (RLtools)** o **TD3 (RLtools)**, así como el copiloto residual de autonomía compartida.

## Requisito

Ya tener configurado el entorno acorde como menciona la guía docker_guide.md de este repositorio.

## 1. Permitir acceso a la interfaz gráfica (en tu terminal local del host)                                                                                                                                                             
                                                                                                                                                                                                                                           
  Si acabas de reiniciar sesión o abrir una terminal nueva en tu sistema, asegúrate de habilitar la salida gráfica X11 para Docker:                                                                                                        

```                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    xhost +local:root                                                                                                                                                                                                                                                                                                                                                                                                                                                    
```
## 2. Terminal 1: abrir el simulador Webots con GUI                                                                                                                                                                                     
                                                                                                                                                                                                                                           
  Entra al contenedor y abre el mundo crazyflie.wbt:                                                                                                                                                                                       
  ```                                                                                                                                                                                                                                         
    docker exec -it webots_rltools_env bash                                                                                                                                                                                                
    webots webots_project/worlds/crazyflie.wbt                                                                                                                                                                                             
  ```
                                                                                                                                                                                                           
> Se abrirá la ventana gráfica de Webots. Como el Crazyflie tiene configurado el controlador en modo <extern>, el simulador quedará a la espera de que el script de entrenamiento se conecte.                                            

## 3. Terminal 2: Lanzar el entrenamiento

Elige entre entrenar con **PPO (Stable-Baselines3)**, **SAC (RLtools en C++)** o **TD3 (RLtools en C++)**:

### Opción A: Entrenamiento con PPO (Stable-Baselines3)

Abre una segunda terminal en tu máquina, entra al contenedor y ejecuta `train_sb3_ppo.py`:

```bash
docker exec -it webots_rltools_env bash
python src/training/train_sb3_ppo.py --timesteps 100000 --max-episode-steps 1000 --corner cone_1
```

* Guarda el modelo en `models/ppo_optimal_pilot.zip` y genera logs en `logs/ppo`.
* Puedes ajustar `--timesteps` (ej. `20000` para una prueba rápida o `100000` para convergencia).

### Opción B: Entrenamiento con SAC (RLtools) y exportación a Header C

Ejecuta el algoritmo Soft Actor-Critic implementado en C++ mediante RLtools:

```bash
docker exec -it webots_rltools_env bash
python src/training/train_rltools_sac.py \
  --steps 100000 \
  --max-episode-steps 1000 \
  --corner cone_1 \
  --export-path models/optimal_pilot_sac_checkpoint.h \
  --log-dir logs/sac
```

* **Compilación C++ en caliente:** RLtools compila automáticamente la interfaz C++ con `g++` y `BLAS`.
* **Exportación a C:** Al finalizar los pasos, exporta la política entrenada directamente como un archivo header C (`models/optimal_pilot_sac_checkpoint.h`), listo para inferencia embebida ultra liviana en el microcontrolador del Crazyflie.

### Opción C: Entrenamiento con TD3 (RLtools wrapper) y exportación a Header C

Ejecuta el algoritmo Twin Delayed DDPG (TD3) con arquitectura de crítico gemelo y actualización demorada del actor, exportando el modelo estático a C:

```bash
docker exec -it webots_rltools_env bash
python src/training/train_rltools_td3.py \
  --steps 100000 \
  --max-episode-steps 1000 \
  --corner cone_1 \
  --export-path models/optimal_pilot_td3_checkpoint.h \
  --log-dir logs/td3
```

* **Política determinista:** Salida acotada en `[-1, 1]` mediante activación Tanh en la capa final del actor.
* **Exportación a C:** Genera `models/optimal_pilot_td3_checkpoint.h` (y su versión `_best.h`), compatible con la especificación de tensores y capas de RLtools para TinyRL.

### Opción D: Entrenamiento del Copiloto Residual Constreñido con PPO (Schaff & Walter, 2020)

Entrena la política residual $\mathbf{a}_r = \pi_r(s_t, \mathbf{a}_h)$ mediante **Constrained PPO** con multiplicador de Lagrange adaptable sobre un operador humano sintético degradado (como el operador ruidoso con $\sigma = 0.50$, idéntico a la configuración del trabajo de Schaff & Walter):

```bash
docker exec -it webots_rltools_env bash
python src/training/train_copilot.py \
  --timesteps 100000 \
  --warmup-steps 10000 \
  --pilot-type noisy \
  --noise-std 0.50 \
  --optimal-backend rltools_td3 \
  --optimal-checkpoint results_TFM/models/optimal_pilot_td3_checkpoint.h \
  --residual-scale 1.0 \
  --init-lambda 1.0 \
  --lr-lambda 0.0003 \
  --target-threshold 0.0 \
  --corner cone_1 \
  --seed 42 \
  --save-path results_TFM/models/copilot_constrained_ppo_noisy.zip \
  --tensorboard-log results_TFM/logs/copilot_constrained_ppo
```

O utilizando el script automatizado con valores por defecto preparados:

```bash
bash scripts/train_copilot.sh
```

* **Aspectos metodológicos clave (Schaff & Walter, 2020):**
  - **Piloto Subrogado Ruidoso (`--pilot-type noisy --noise-std 0.50`):** Simula el temblor o degradación motriz del operador humano inyectando perturbaciones gaussianas $\mathcal{N}(0, 0.50^2)$ sobre el espacio de acción normalizado $[-1, 1]$.
  - **Fase de Calentamiento (*Warm-up*, `--warmup-steps 10000`):** Durante los primeros $10\,000$ pasos de entrenamiento, la acción residual se anula forzadamente ($\mathbf{a}_r = 0, \beta = 0$). Esto permite que la red del Crítico aprenda la función de valor $V(s, \mathbf{a}_h)$ sobre las trayectorias no asistidas del operador antes de que el Actor residual comience a emitir correcciones.
  - **Multiplicador de Lagrange Adaptable (`--init-lambda 1.0 --lr-lambda 0.0003`):** Modula dinámicamente la penalización por intervención según el estado de riesgo. Si el cuadricóptero se acerca a muros o piso, $\lambda$ se ajusta para salvaguardar la integridad del vehículo; cuando la maniobra del piloto es segura, el término dual fuerza a que la norma residual tienda a cero ($\|\mathbf{a}_r\|_2 \to 0$, principio de mínima intervención).
  - **Checkpoint base óptimo (`--optimal-checkpoint` y `--optimal-backend`):** Permite utilizar como política base TD3 (`rltools_td3`), SAC (`rltools_sac`) o PPO (`sb3_ppo`).

### Opción E: Entrenamiento del Copiloto Residual No Constreñido (Línea base unconstrained)

Para fines comparativos y de ablación, se puede entrenar un copiloto residual estándar mediante PPO/SAC convencional penalizando la norma de intervención de manera estática con factor fijo $\lambda$:

```bash
docker exec -it webots_rltools_env bash
python src/training/train_residual_copilot.py \
  --algorithm ppo \
  --pilot-type noisy \
  --noise-std 0.50 \
  --timesteps 100000 \
  --lambda-intervention 0.10 \
  --residual-scale 1.0 \
  --save-path models/copilot_residual_ppo_noisy.zip \
  --tensorboard-log results_TFM/logs/copilot_unconstrained
```

---

## 4. Evaluación del Piloto Óptimo TD3 y Gráfico de Trayectoria

Una vez concluido el entrenamiento de TD3, evalúa la política aprendida sobre 100 episodios independientes utilizando una semilla maestra desacoplada (`--seed 1000`) y genera automáticamente el gráfico de trayectorias 2D:

```bash
docker exec -it webots_rltools_env bash
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --backend rltools_td3 \
  --checkpoint models/optimal_pilot_td3_checkpoint.h \
  --episodes 100 \
  --seed 1000 \
  --corner cone_1 \
  --save-csv logs/eval_optimal_td3_100ep.csv \
  --plot-trajectory-png logs/trayectoria_optimal_td3_100ep.png
```

* **Salidas generadas:**
  - `logs/eval_optimal_td3_100ep.csv`: Resultados detallados de cada episodio (recompensa, longitud, éxito, distancia final y semilla única asignada).
  - `logs/trayectoria_optimal_td3_100ep.png`: Gráfico 2D con las 100 trayectorias superpuestas en la habitación de $2 \times 2\,\text{m}$, con el punto de partida `[0, 0]`, los conos y las paredes.

---

## 5. Cómo controlar la visualización dentro de Webots

Una vez que el script se conecte, la simulación arrancará automáticamente:

* **Velocidad de simulación (barra superior de Webots):**
  - **Play (Tiempo real):** Te permite observar la física del cuadricóptero, la inclinación (pitch/roll) y la altitud a velocidad natural.
  - **Fast / Run:** Acelera la simulación computacional al máximo de tu CPU/GPU. Verás al dron moverse en cámara rápida de un intento a otro, lo que permite entrenar más rápido mientras sigues viendo la animación 3D.
* **Qué esperar en los episodios:**
  - En cada reset el dron despegará y se estabilizará a 0.50 m.
  - En los primeros episodios explorará con movimientos más ruidosos; si choca contra las paredes o cae, el episodio se reiniciará inmediatamente.
  - Conforme avanza el entrenamiento, consolidará trayectorias directas hacia `cone_1` (+0.9, +0.9).

---

## 6. Modo rápido sin interfaz gráfica (Headless)

Si prefieres entrenar o evaluar a máxima velocidad computacional sin gastar ciclos de GPU ni dibujar ventanas 3D:

```bash
# Terminal 1: Iniciar Webots headless
docker exec -it webots_rltools_env bash
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt
```
Y en la Terminal 2 ejecutas el script de entrenamiento o evaluación elegido (`train_rltools_td3.py` o `metrics.py`).

O en una **sola terminal** en segundo plano:

```bash
docker exec -it webots_rltools_env bash
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt > /tmp/webots.log 2>&1 &
python src/training/train_rltools_td3.py --steps 100000 --export-path models/optimal_pilot_td3_checkpoint.h
python src/evaluation/metrics.py --pilot-type optimal --backend rltools_td3 --checkpoint models/optimal_pilot_td3_checkpoint.h --episodes 100 --seed 1000 --save-csv logs/eval_optimal_td3_100ep.csv --plot-trajectory-png logs/trayectoria_optimal_td3_100ep.png
pkill -f webots-bin
```

---

## 7. Automatización Multisemilla (5 Semillas para TD3, SAC y PPO)

Para dar respuesta rigurosa al revisor y evaluar la convergencia estadística sin sesgo de inicialización aleatoria, dispones del script automatizado [`scripts/train_all_seeds.sh`](file:///home/ana/001_TFM_proyecto/0001-webots-rl/scripts/train_all_seeds.sh):

```bash
# Ejecutar la campaña completa (TD3, SAC y PPO x 5 semillas = 15 entrenamientos y 15 evaluaciones):
bash scripts/train_all_seeds.sh

# O ejecutar por algoritmo individual:
bash scripts/train_all_seeds.sh td3
bash scripts/train_all_seeds.sh sac
bash scripts/train_all_seeds.sh ppo
```

Todos los modelos, logs de TensorBoard y CSVs de evaluación se almacenan de forma organizada en `results_TFM/` con la nomenclatura `_seed_<S>`.

---

## 8. (Opcional) Ver las métricas en TensorBoard

Para monitorear curvas de recompensa, longitud de episodios y pérdidas del algoritmo en tu navegador:

```bash
# Para monitorear TD3 (RLtools):
tensorboard --logdir ./logs/td3

# Para monitorear SAC (RLtools):
tensorboard --logdir ./logs/sac

# Para monitorear PPO:
tensorboard --logdir ./logs/ppo

# Para monitorear el Copiloto Residual:
tensorboard --logdir ./logs/copilot

# O para comparar todos los entrenamientos a la vez:
tensorboard --logdir ./logs
```

Luego abre en el navegador: [http://localhost:6006](http://localhost:6006)
