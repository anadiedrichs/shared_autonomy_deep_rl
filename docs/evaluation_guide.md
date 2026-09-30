# Evaluation Guide

Guía para la evaluación cuantitativa de los pilotos (óptimo, heurístico, ruidoso y con retardo) en el entorno Webots, y el cálculo de métricas estadísticas análogas a las tablas 4.1 a 4.4 de la tesis (Diedrichs, 2025).

---

## Requisitos

Tener el contenedor Docker en ejecución con el entorno de simulación configurado (según `docs/docker_guide.md` y `docs/training_guide.md`).

---

## 1. Permitir acceso a la interfaz gráfica (en tu terminal local del host)

Si abres una terminal nueva o reiniciaste sesión, habilita la salida gráfica X11 para Docker:

```bash
xhost +local:root
```

---

## 2. Terminal 1: Abrir el simulador Webots con GUI

Entra al contenedor y abre el mundo `crazyflie.wbt`:

```bash
docker exec -it webots_rltools_env bash
webots webots_project/worlds/crazyflie.wbt
```

> **Nota:** La ventana gráfica de Webots se abrirá y quedará a la espera de la conexión del controlador externo `<extern>`.

---

## 3. Terminal 2: Ejecutar la evaluación de pilotos

Abre una segunda terminal en tu máquina, entra al contenedor y ejecuta `src/evaluation/metrics.py`.

El script soporta evaluar los diferentes tipos de pilotos mediante el argumento `--pilot-type`:

### A. Evaluar el Piloto Óptimo entrenado (PPO)

Evalúa la política aprendida de manera **determinista** (sin ruido de exploración):

```bash
docker exec -it webots_rltools_env bash
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --checkpoint models/ppo_optimal_pilot.zip \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_ppo_optimal.csv
```

### B. Evaluar el Piloto Heurístico (Benchmark / Referencia Geométrica)

Evalúa el controlador proporcional en lazo cerrado para contrastar contra el piloto óptimo:

```bash
python src/evaluation/metrics.py \
  --pilot-type heuristic \
  --heuristic-mode proportional \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_heuristic.csv
```

### C. Evaluar el Piloto Ruidoso (NoisyPilot - Simulación de temblor o error humano)

Inyecta perturbaciones gaussianas $\mathcal{N}(0, \sigma^2)$ sobre las acciones continuas del piloto óptimo:

```bash
python src/evaluation/metrics.py \
  --pilot-type noisy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --sigma 0.15 \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_noisy_sigma015.csv
```

### D. Evaluar el Piloto con Retardo (LaggyPilot - Simulación de latencia de reacción)

Introduce inercia temporal repitiendo la última acción con probabilidad $p_{\text{repeat}}$:

```bash
python src/evaluation/metrics.py \
  --pilot-type laggy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --p-repeat 0.25 \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_laggy_p025.csv
```

---

## 4. Opciones y parámetros de línea de comandos

| Argumento | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--pilot-type` | `str` | `optimal` | Tipo de piloto: `optimal`, `heuristic`, `noisy`, `laggy`. |
| `--checkpoint` | `str` | `models/ppo_optimal_pilot.zip` | Ruta al archivo del modelo entrenado (`.zip` o `.h`). |
| `--backend` | `str` | `sb3_ppo` | Backend para OptimalPilot: `sb3_ppo`, `rltools_sac`, `dummy`. |
| `--heuristic-mode` | `str` | `proportional` | Modalidad heurística: `proportional` u `open_loop`. |
| `--episodes` | `int` | `10` | Cantidad de episodios a evaluar. |
| `--corner` | `str` | `cone_1` | Cono objetivo en Webots (`cone_1`, `cone_2`, `cone_3`, `cone_4`). |
| `--max-episode-steps`| `int` | `1000` | Límite máximo de pasos por episodio antes de truncar. |
| `--sigma` | `float` | `0.1` | Desviación estándar del ruido para `NoisyPilot`. |
| `--p-repeat` | `float` | `0.2` | Probabilidad de repetición de acción para `LaggyPilot`. |
| `--save-csv` | `str` | `logs/eval_results.csv` | Ruta destino para guardar los resultados detallados por episodio. |
| `--verbose` | `int` | `1` | `0`: silencioso, `1`: resumen por episodio y global, `2`: paso a paso. |

---

## 5. Salida de métricas y formato CSV

### En la terminal

Al finalizar la corrida, se imprime un resumen estadístico:

```text
=================================================================
 RESUMEN DE EVALUACIÓN (Patrón Tesis 2025):
  Tasa de Éxito   (Success Rate) : 95.0% (19/20)
  Tasa de Caídas  (Crash Rate)   : 5.0% (1/20)
  Tasa de Tiempo  (Timeout Rate) : 0.0% (0/20)
  Recompensa Media               : 74.20 ± 8.15
  Longitud Media (pasos)         : 48.3 ± 4.2
  Distancia Final Media          : 0.124m ± 0.015m
=================================================================
```

### En el archivo CSV (`logs/...csv`)

El archivo CSV generado contiene las siguientes columnas por episodio:

* `episode`: Número secuencial del episodio.
* `reward`: Recompensa acumulada obtenida en el episodio.
* `length`: Cantidad de pasos ejecutados antes de terminar.
* `success`: Booleano (`True`/`False`), indica si alcanzó el cono con altitud segura $> 0.20\,\text{m}$.
* `termination`: Motivo de finalización (`goal`, `fall`, `out_of_bounds`, `timeout`).
* `final_distance`: Distancia euclidiana mínima en metros al centro del cono en el último paso.

Este archivo se puede cargar directamente con `pandas` para análisis o generación de gráficos en Jupyter notebooks:

```python
import pandas as pd
df = pd.read_csv("logs/eval_ppo_optimal.csv")
print(f"Success rate: {df['success'].mean() * 100:.1f}%")
```
