# Training guide

El how-to para entrenar y visualizar el entrenamiento en webots.

En este ejemplo observamos como entrenar un piloto óptimo con PPO.

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

Elige entre entrenar con **PPO (Stable-Baselines3)** o con **SAC (RLtools en C++)**:

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

### Opción C: Entrenamiento del Copiloto Residual (Autonomía Compartida)

Entrena la política residual $\pi_r(s, a_h)$ manteniendo congelado a un piloto humano o sintético degradado (`noisy` o `laggy`), optimizando la recompensa de seguridad *goal-agnostic* (Schaff & Walter, 2020):

```bash
docker exec -it webots_rltools_env bash
python src/training/train_residual_copilot.py \
  --algorithm ppo \
  --pilot-type noisy \
  --noise-std 0.20 \
  --timesteps 100000 \
  --lambda-intervention 0.10 \
  --residual-scale 1.0 \
  --save-path models/copilot_residual_ppo_noisy.zip \
  --tensorboard-log logs/copilot
```

* **Parámetros clave:**
  - `--algorithm`: Algoritmo de RL (`ppo` o `sac`).
  - `--pilot-type`: Piloto congelado a asistir (`noisy`, `laggy`, `heuristic`, `optimal`).
  - `--lambda-intervention`: Penalización por magnitud $\|a_r\|^2$ (por defecto `0.10`); cuanto mayor es, más se restringe la intervención del copiloto.
  - `--residual-scale`: Factor de atenuación para limitar la autoridad del copiloto sobre los mandos del piloto (en $(0, 1]$).

---

## 4. Cómo controlar la visualización dentro de Webots

Una vez que el script se conecte, la simulación arrancará automáticamente:

* **Velocidad de simulación (barra superior de Webots):**
  - **Play (Tiempo real):** Te permite observar la física del cuadricóptero, la inclinación (pitch/roll) y la altitud a velocidad natural.
  - **Fast / Run:** Acelera la simulación computacional al máximo de tu CPU/GPU. Verás al dron moverse en cámara rápida de un intento a otro, lo que permite entrenar más rápido mientras sigues viendo la animación 3D.
* **Qué esperar en los episodios:**
  - En cada reset el dron despegará y se estabilizará a 0.50 m.
  - En los primeros episodios explorará con movimientos más ruidosos; si choca contra las paredes o cae, el episodio se reiniciará inmediatamente.
  - Conforme avanza el entrenamiento, consolidará trayectorias directas hacia `cone_1` (+0.9, +0.9).

---

## 5. Entrenamiento rápido en modo sin interfaz gráfica (Headless)

Si prefieres entrenar a máxima velocidad computacional sin gastar ciclos de GPU ni dibujar ventanas 3D:

```bash
docker exec -it webots_rltools_env bash
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt
```
Y en la Terminal 2 ejecutas el script de entrenamiento elegido (PPO o SAC).

---

## 6. (Opcional) Ver las métricas en TensorBoard

Para monitorear curvas de recompensa, longitud de episodios y pérdidas del algoritmo en tu navegador:

```bash
# Para monitorear PPO:
tensorboard --logdir ./logs/ppo

# Para monitorear SAC (RLtools):
tensorboard --logdir ./logs/sac

# Para monitorear el Copiloto Residual:
tensorboard --logdir ./logs/copilot

# O para comparar todos los entrenamientos a la vez:
tensorboard --logdir ./logs
```

Luego abre en el navegador: [http://localhost:6006](http://localhost:6006)
