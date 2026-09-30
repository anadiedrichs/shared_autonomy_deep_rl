# Evaluation Guide

Guía para la evaluación cuantitativa de los pilotos (óptimo, heurístico, ruidoso y con retardo) en el entorno Webots, y el cálculo de métricas estadísticas análogas a las tablas 4.1 a 4.4 de la tesis (Diedrichs, 2025).

---

## Requisitos

Tener el contenedor Docker en ejecución con el entorno de simulación configurado (según `docs/docker_guide.md` y `docs/training_guide.md`).

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

Desactiva el motor OpenGL y corre la física al máximo rendimiento de la CPU. **20 episodios se evalúan en apenas 10 a 15 segundos**:

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

# 3. Ejecutar la evaluación de pilotos (se conectará en 1 segundo)
python src/evaluation/metrics.py --pilot-type optimal --checkpoint models/ppo_optimal_pilot.zip --episodes 20 --corner cone_1 --save-csv logs/eval_ppo_optimal.csv

# 4. Al finalizar la evaluación, detener el proceso de Webots en segundo plano:
pkill -f webots-bin
```

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

### E. Evaluar el Piloto Asistido (AssistedPilot con Copiloto Residual)

Evalúa la composición aditiva $a = \text{clip}(a_h + a_r, -1, 1)$ aplicando el copiloto entrenado sobre un piloto degradado o con retardo:

```bash
python src/evaluation/metrics.py \
  --pilot-type assisted \
  --base-pilot-type noisy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --sigma 0.20 \
  --copilot-checkpoint models/copilot_residual_ppo_noisy.zip \
  --residual-scale 1.0 \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_assisted_noisy.csv
```

---

## 4. Opciones y parámetros de línea de comandos

| Argumento | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--pilot-type` | `str` | `optimal` | Tipo de piloto: `optimal`, `heuristic`, `noisy`, `laggy`, `assisted`. |
| `--checkpoint` | `str` | `models/ppo_optimal_pilot.zip` | Ruta al archivo del modelo entrenado (`.zip` o `.h`). |
| `--backend` | `str` | `sb3_ppo` | Backend para OptimalPilot: `sb3_ppo`, `rltools_sac`, `dummy`. |
| `--heuristic-mode` | `str` | `proportional` | Modalidad heurística: `proportional` u `open_loop`. |
| `--base-pilot-type` | `str` | `noisy` | Piloto base asistido si `pilot-type=assisted` (`noisy`, `laggy`, `optimal`). |
| `--copilot-checkpoint` | `str` | `None` | Ruta al modelo del copiloto residual (`.zip` o `.h`). |
| `--copilot-backend` | `str` | `auto` | Backend del copiloto (`auto`, `sb3_ppo`, `sb3_sac`, `rltools_sac`, `dummy`). |
| `--residual-scale` | `float` | `1.0` | Factor de escala para limitar la intervención residual $a_r$. |
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

---

## 6. Comparación y generación de reportes cuantitativos (`compare_pilots.py`)

Para agregar y comparar sistemáticamente los resultados de múltiples pilotos (por ejemplo, contrastar un piloto no asistido frente a su versión con copiloto residual):

```bash
docker exec -it webots_rltools_env bash
python src/evaluation/compare_pilots.py \
  --csv-files \
    logs/eval_heuristic.csv \
    logs/eval_ppo_optimal.csv \
    logs/eval_noisy_sigma015.csv \
    logs/eval_assisted_noisy.csv \
  --labels \
    "Heurístico" \
    "Óptimo PPO" \
    "Noisy (solo)" \
    "Noisy + Copiloto" \
  --output-dir logs/
```

### Artefactos generados:
1. **Tabla CSV agregada (`logs/pilots_comparison_summary.csv`):** Tasa de éxito, tasa de colisión, duración y distancias.
2. **Tabla Markdown (`logs/pilots_comparison_summary.md`):** Formateada para inclusión directa en reportes y tesis.
3. **Gráficos PNG (`logs/pilots_success_vs_collision.png` y `logs/pilots_rewards_and_lengths.png`):** Figuras en alta resolución listas para publicación.

### 7. Análisis interactivo en Jupyter Notebook

Para explorar trayectorias espaciales 2D/3D en la habitación de $2 \times 2\text{ m}$ y curvas de activación del residual, abre el notebook:
`notebooks/01_pilots_and_copilot_analysis.ipynb`
