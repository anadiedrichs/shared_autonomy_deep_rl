# Evaluation Guide

Guía para la evaluación cuantitativa de los pilotos (óptimo PPO, SAC y TD3; heurístico; ruidoso y con retardo) en el entorno Webots, con registro de trayectorias espaciales 2D y cálculo de métricas estadísticas reproducibles bajo el protocolo riguroso de 100 episodios y semilla maestra desacoplada.

---

## Requisitos

Tener el contenedor Docker en ejecución con el entorno de simulación configurado (según [`docs/docker_guide.md`](docker_guide.md) y [`docs/training_guide.md`](training_guide.md)).

---

## 1. Acceso gráfico (Opcional, solo si usas modo con GUI)

Si vas a abrir la ventana 3D de Webots en tu monitor, habilita la salida gráfica X11 en tu máquina host:

```bash
xhost +local:root
```

*(Si vas a evaluar en **modo headless / sin interfaz gráfica**, puedes saltarte este paso por completo).*

---

## 2. Iniciar el simulador Webots (Elige una modalidad)

Elige entre visualizar el vuelo en 3D o evaluar a máxima velocidad sin gráficos:

### Modalidad A: Con interfaz gráfica 3D (GUI)

Permite observar visualmente la actitud de vuelo del Crazyflie en tu monitor:

```bash
docker exec -it webots_rltools_env bash
webots webots_project/worlds/crazyflie.wbt
```

> **Nota:** La ventana gráfica de Webots se abrirá y quedará a la espera de la conexión del controlador `<extern>`.

### Modalidad B: Modo sin interfaz gráfica (Headless / Ultra rápido)

Desactiva el motor OpenGL y corre la física al máximo rendimiento de la CPU. **100 episodios se evalúan en apenas 40 a 60 segundos**:

```bash
docker exec -it webots_rltools_env bash
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt
```

**Parámetros explicados:**
* `xvfb-run -a`: Crea un servidor virtual de pantalla en memoria RAM (*X Virtual Framebuffer*), permitiendo que Webots corra sin necesidad de pantalla física ni permisos `xhost`.
* `--batch`: Modo no interactivo (evita diálogos emergentes o ventanas de confirmación).
* `--mode=fast`: Calcula la física ODE a máxima velocidad computacional (sin esperar al reloj de tiempo real).
* `--no-rendering`: **Desactiva por completo el dibujo y renderizado 3D**, ahorrando GPU y ciclos de CPU.

---

### Alternativa: Ejecutar todo en una sola terminal (Webots en segundo plano)

Si prefieres no usar dos terminales separadas, puedes iniciar Webots en segundo plano (*background*) y correr la evaluación inmediatamente en la misma consola:

```bash
# 1. Entrar al contenedor
docker exec -it webots_rltools_env bash

# 2. Iniciar Webots headless en segundo plano silenciando logs
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt > /tmp/webots.log 2>&1 &

# 3. Ejecutar la evaluación de pilotos (se conectará automáticamente)
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --backend rltools_td3 \
  --checkpoint models/optimal_pilot_td3_checkpoint.h \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_optimal_td3_100ep.csv \
  --plot-trajectory-png logs/trayectoria_optimal_td3_100ep.png

# 4. Al finalizar la evaluación, detener el proceso de Webots en segundo plano:
pkill -f webots-bin
```

---

## 3. Protocolo de Evaluación Estadística Rigurosa

Para garantizar solidez metodológica y evitar sesgos experimentales:

1. **Tamaño muestral ($N = 100$ episodios):**
   Evaluar 100 episodios reduce el error estándar de la media a $\frac{\sigma}{\sqrt{100}} = 0.10\,\sigma$, permitiendo calcular intervalos de confianza del 95% ($\pm 1.96 \cdot SEM$) formalmente defendibles ante el tribunal.
2. **Semilla maestra desacoplada (`--seed 1000`):**
   Los entrenamientos utilizaron las semillas `7` (pilotos óptimos) y `42` (copiloto residual). Para asegurar que la evaluación se realice sobre secuencias pseudoaleatorias no vistas (*out-of-distribution*), se utiliza una semilla inicial distinta (por defecto `1000`).
3. **Generación determinista de semillas por episodio:**
   A partir de la semilla maestra, el evaluador genera determinísticamente una lista de 100 semillas únicas (`episode_seeds`). Cada episodio $i$ recibe su propia semilla, la cual se propaga al entorno (`env.reset(seed=ep_seed)`) y al piloto (`pilot.reset(seed=ep_seed)`). De este modo, las perturbaciones (ruido de temblor o retardos de latencia) son variadas entre episodios, pero **estrictamente idénticas y reproducibles** al comparar el piloto solo vs. asistido.
4. **Gráficos de trayectoria 2D (`--plot-trajectory-png`):**
   Genera una imagen PNG de alta resolución ($200\text{ dpi}$) donde se superponen las trayectorias de los 100 vuelos en el recinto de $2 \times 2\text{ m}$, con transparencia adaptativa ($\alpha$), marcando el inicio en `[0, 0]`, las paredes, los 4 conos y el objetivo final.

---

## 4. Comandos de Evaluación por Perfil de Piloto

Abre una segunda terminal en tu máquina, entra al contenedor y ejecuta `src/evaluation/metrics.py`:

```bash
docker exec -it webots_rltools_env bash
```

### A. Piloto Óptimo TD3 (RLtools en C++)
Evalúa la política determinista continua entrenada con Twin Delayed DDPG exportada como cabecera C:

```bash
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --backend rltools_td3 \
  --checkpoint models/optimal_pilot_td3_checkpoint.h \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_optimal_td3_100ep.csv \
  --plot-trajectory-png logs/trayectoria_optimal_td3_100ep.png
```

### B. Piloto Óptimo SAC (RLtools en C++)
Evalúa la política continua entrenada con Soft Actor-Critic en RLtools exportada como cabecera C:

```bash
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --backend rltools_sac \
  --checkpoint models/optimal_pilot_sac_checkpoint.h \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_optimal_sac_100ep.csv \
  --plot-trajectory-png logs/trayectoria_optimal_sac_100ep.png
```

### C. Piloto Óptimo PPO (Stable-Baselines3)
Evalúa la política continua aprendida con PPO de manera determinista:

```bash
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --backend sb3_ppo \
  --checkpoint models/ppo_optimal_pilot.zip \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_optimal_ppo_100ep.csv \
  --plot-trajectory-png logs/trayectoria_optimal_ppo_100ep.png
```

### D. Piloto Heurístico (Benchmark / Referencia Geométrica)
Evalúa el controlador proporcional en lazo cerrado para contrastar contra el piloto óptimo:

```bash
python src/evaluation/metrics.py \
  --pilot-type heuristic \
  --heuristic-mode proportional \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_heuristic_100ep.csv \
  --plot-trajectory-png logs/trayectoria_heuristic_100ep.png
```

### E. Operador Ruidoso Solo (NoisyPilot - Temblor o imprecisión humana)
Inyecta perturbaciones gaussianas $\mathcal{N}(0, \sigma^2)$ sobre las acciones del piloto óptimo:

```bash
python src/evaluation/metrics.py \
  --pilot-type noisy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --sigma 0.30 \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_noisy_solo_100ep.csv \
  --plot-trajectory-png logs/trayectoria_noisy_solo_100ep.png
```

### F. Operador Ruidoso Asistido por Copiloto Residual (Autonomía Compartida)
Evalúa la composición aditiva $a = \text{clip}(a_h + a_r, -1, 1)$ aplicando el copiloto entrenado sobre el operador degradado:

```bash
python src/evaluation/metrics.py \
  --pilot-type assisted \
  --base-pilot-type noisy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --sigma 0.30 \
  --copilot-checkpoint models/best_copilot_ppo_noisy/best_model.zip \
  --copilot-backend sb3_ppo \
  --residual-scale 1.0 \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_noisy_asistido_ppo_100ep.csv \
  --plot-trajectory-png logs/trayectoria_noisy_asistido_ppo_100ep.png
```

### G. Operador con Retardo Solo (LaggyPilot - Latencia o pérdida de paquetes)
Introduce retraso temporal omitiendo comandos con probabilidad $p_{\mathrm{drop}}$:

```bash
python src/evaluation/metrics.py \
  --pilot-type laggy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --p-repeat 0.05 \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_laggy_solo_100ep.csv \
  --plot-trajectory-png logs/trayectoria_laggy_solo_100ep.png
```

### H. Operador con Retardo Asistido por Copiloto Residual
```bash
python src/evaluation/metrics.py \
  --pilot-type assisted \
  --base-pilot-type laggy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --p-repeat 0.05 \
  --copilot-checkpoint models/best_copilot_ppo_laggy/best_model.zip \
  --copilot-backend sb3_ppo \
  --residual-scale 1.0 \
  --episodes 100 \
  --seed 1000 \
  --save-csv logs/eval_laggy_asistido_ppo_100ep.csv \
  --plot-trajectory-png logs/trayectoria_laggy_asistido_ppo_100ep.png
```

---

## 5. Tabla de Parámetros de Línea de Comandos

| Argumento | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--pilot-type` | `str` | `optimal` | Tipo de piloto: `optimal`, `heuristic`, `noisy`, `laggy`, `assisted`. |
| `--checkpoint` | `str` | `models/ppo_optimal_pilot.zip` | Ruta al archivo del modelo entrenado (`.zip` o `.h`). |
| `--backend` | `str` | `sb3_ppo` | Backend para OptimalPilot: `sb3_ppo`, `rltools_sac`, `rltools_td3`, `dummy`. |
| `--heuristic-mode` | `str` | `proportional` | Modalidad heurística: `proportional` u `open_loop`. |
| `--base-pilot-type` | `str` | `noisy` | Piloto base asistido si `pilot-type=assisted` (`noisy`, `laggy`, `optimal`). |
| `--copilot-checkpoint` | `str` | `None` | Ruta al modelo del copiloto residual (`.zip` o `.h`). |
| `--copilot-backend` | `str` | `auto` | Backend del copiloto (`auto`, `sb3_ppo`, `sb3_sac`, `rltools_sac`, `rltools_td3`, `dummy`). |
| `--residual-scale` | `float` | `1.0` | Factor de escala para limitar la intervención residual $a_r$. |
| `--episodes` | `int` | `100` | Cantidad de episodios a evaluar (100 recomendados para significancia estadística). |
| `--seed` | `int` | `1000` | Semilla maestra de evaluación (independiente de las semillas 7 y 42 de entrenamiento). Genera determinísticamente las semillas únicas de cada episodio. |
| `--corner` | `str` | `cone_1` | Cono objetivo en Webots (`cone_1`, `cone_2`, `cone_3`, `cone_4`). |
| `--max-episode-steps`| `int` | `1000` | Límite máximo de pasos por episodio antes de truncar. |
| `--sigma` | `float` | `0.1` | Desviación estándar del ruido para `NoisyPilot`. |
| `--p-repeat` | `float` | `0.2` | Probabilidad de repetición/omisión de acción para `LaggyPilot`. |
| `--save-csv` | `str` | `logs/eval_results.csv` | Ruta destino para guardar los resultados detallados por episodio. |
| `--plot-trajectory-png`| `str` | `None` | Ruta destino para guardar el gráfico PNG de trayectorias 2D superpuestas en el recinto. |
| `--log-trajectory` | `bool` | `False` | Si se activa, guarda los puntos `[x, y, z]` en el diccionario de resultados. |
| `--verbose` | `int` | `1` | `0`: silencioso, `1`: resumen por episodio y global, `2`: paso a paso. |

---

## 6. Salida de Métricas y Formato CSV

### En la terminal

Al finalizar la corrida, se imprime un resumen estadístico detallado:

```text
=================================================================
 RESUMEN DE EVALUACIÓN (Patrón Tesis 2025):
  Episodios Evaluados            : 100
  Semilla Maestra de Evaluación  : 1000
  Tasa de Éxito   (Success Rate) : 96.0% (96/100)
  Tasa de Caídas  (Crash Rate)   : 4.0% (4/100)
  Tasa de Tiempo  (Timeout Rate) : 0.0% (0/100)
  Recompensa Media               : 75.42 ± 6.18
  Longitud Media (pasos)         : 47.1 ± 3.8
  Distancia Final Media          : 0.118m ± 0.012m
  Altitud Final Media            : 498.2mm ± 5.4mm
=================================================================
```

### En el archivo CSV (`logs/...csv`)

El archivo CSV generado contiene las siguientes columnas trazables por episodio:

* `episode`: Número secuencial del episodio ($1$ a $N$).
* `seed`: Semilla pseudoaleatoria única asignada a ese episodio.
* `reward`: Recompensa acumulada obtenida en el episodio.
* `length`: Cantidad de pasos ejecutados antes de terminar.
* `success`: Booleano (`True`/`False`), indica si alcanzó el cono con altitud segura $> 0.20\,\text{m}$.
* `termination`: Motivo de finalización (`goal`, `fall`, `out_of_bounds`, `timeout`).
* `final_distance`: Distancia euclidiana mínima en metros al centro del cono en el último paso.
* `final_altitude_mm`: Altitud final del dron en milímetros.

---

## 7. Comparación Global y Generación de Reportes (`compare_pilots.py`)

Para agregar y comparar sistemáticamente los resultados de múltiples pilotos con sus intervalos de error estándar ($SEM$):

```bash
docker exec -it webots_rltools_env bash
python src/evaluation/compare_pilots.py \
  --csv-files \
    logs/eval_heuristic_100ep.csv \
    logs/eval_optimal_ppo_100ep.csv \
    logs/eval_optimal_sac_100ep.csv \
    logs/eval_optimal_td3_100ep.csv \
    logs/eval_noisy_solo_100ep.csv \
    logs/eval_noisy_asistido_ppo_100ep.csv \
  --labels \
    "Heurístico" \
    "Óptimo PPO (SB3)" \
    "Óptimo SAC (RLtools)" \
    "Óptimo TD3 (RLtools)" \
    "Ruidoso Solo" \
    "Ruidoso + Copiloto PPO" \
  --output-dir logs/
```

### Artefactos generados:
1. **Tabla CSV agregada (`logs/pilots_comparison_summary.csv`):** Tasas de éxito, caídas, medias y error estándar ($SEM$) de recompensa y longitud.
2. **Tabla Markdown (`logs/pilots_comparison_summary.md`):** Formateada para inserción directa en informes y tesis.
3. **Gráficos PNG:**
   - `logs/pilots_success_vs_collision.png`: Gráfico de barras comparativo de éxito vs. colisión.
   - `logs/pilots_rewards_and_lengths.png`: Gráfico con barras de error estándar ($\pm SEM$).
