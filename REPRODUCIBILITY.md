# Guía de reproducibilidad de resultados / Reproducibility guide

**Trabajo de fin de máster (TFM):** *Autonomía Compartida para UAVs: Desarrollo Sim-to-Real con Webots y RLtools*  
**Autora:** Ana Laura Diedrichs  
**Institución:** Universidad Europea de Madrid  
**Repositorio:** `shared_autonomy_deep_rl`

---

## Objetivo de esta guía

Este documento proporciona las instrucciones completas, transparentes y sistemáticas para **reproducir de manera exacta todas las tablas numéricas, métricas cuantitativas y figuras comparativas** presentadas en la memoria del TFM (Capítulo 4: *Desarrollo y Resultados* y Capítulo 5: *Discusión*).

Para adaptarse a la disponibilidad de tiempo y recursos computacionales del tribunal y evaluadores, la reproducibilidad se estructura en **tres niveles independientes**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ NIVEL 1: REPRODUCIBILIDAD OFFLINE INMEDIATA (< 1 minuto)                     │
│ • No requiere Webots, GPU ni display gráfico.                                │
│ • Lee directamente los datos experimentales consolidados en results_TFM/.   │
│ • Regenera en consola, Markdown y LaTeX las Tablas 4.2 a 4.5 y sus gráficos. │
├──────────────────────────────────────────────────────────────────────────────┤
│ NIVEL 2: EVALUACIÓN DE INFERENCIA EN WEBOTS (~ 5 a 15 minutos)              │
│ • Carga los modelos pre-entrenados alojados en results_TFM/models/.          │
│ • Ejecuta baterías de evaluación deterministas y estocásticas en simulación. │
│ • Verifica físicamente la tasa de éxito, colisiones y maniobras de escape.   │
├──────────────────────────────────────────────────────────────────────────────┤
│ NIVEL 3: RE-ENTRENAMIENTO MULTISEMILLA DESDE CERO (Varias horas)             │
│ • Entrena políticas continuas completas (TD3, SAC, PPO, Copiloto).           │
│ • Aplica exactamente los hiperparámetros auditados en los Anexos 1 a 5.      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Matriz de trazabilidad: memoria $\leftrightarrow$ código y datos

A continuación se detalla la correspondencia exacta entre cada artefacto publicado en la memoria escrita y los scripts y datos que lo sustentan:

| Elemento en memoria | Descripción | Script de reproducción | Fuente de datos (`results_TFM/`) |
| :--- | :--- | :--- | :--- |
| **Tabla 4.1** | Configuraciones de evaluación de pilotos degradados | `scripts/reproduce_results.py --table 4.1` | Parámetros canónicos del marco teórico |
| **Tabla 4.2** | Comparativa cuantitativa: Pilotos Óptimos PPO vs. SAC vs. TD3 | `scripts/reproduce_results.py --table 4.2` | `results_TFM/logs/td3/eval_optimal_*.csv`, `results_TFM/logs/sac/` |
| **Tabla 4.3** | Desempeño de pilotos sintéticos degradados (Noisy, Laggy, StandBy) | `scripts/reproduce_results.py --table 4.3` | `results_TFM/logs/degraded/` |
| **Tabla 4.4** | Desempeño del copiloto residual asistiendo a piloto ruidoso ($\sigma=0.30$) | `scripts/reproduce_results.py --table 4.4` | `results_TFM/logs/copilot/` |
| **Tabla 4.5** | Desempeño del copiloto residual asistiendo a piloto lento ($p_{\text{drop}}$) | `scripts/reproduce_results.py --table 4.5` | `results_TFM/logs/copilot/` |
| **Figuras de trayectorias** | Recorrido espacial $XY$ en recinto de $2\times 2\,\text{m}$ | `scripts/reproduce_results.py --plots` | `results_TFM/logs/*/*.png` |
| **Anexos 1 a 5** | Hiperparámetros de entrenamiento (TD3, SAC, PPO, Copiloto) | Inspección en código | `src/training/`, `models/*_hyperparameters.json` |

---

## Nivel 1: reproducción offline inmediata (< 1 min)

Este nivel permite al evaluador validar todas las afirmaciones numéricas del texto sin necesidad de compilar código C++, configurar Webots ni disponer de GPU.

### Opción A: script automatizado por consola
Ejecutar desde el directorio raíz del repositorio:
```bash
python3 scripts/reproduce_results.py
```

**Salidas generadas:**
1. Tablas en consola con formato ASCII/Markdown idénticas a las de la memoria.
2. Archivos LaTeX (`.tex`) listos para compilar en `./reproduced_results/tables/`.
3. Gráficos comparativos regenerados en `./reproduced_results/figures/`:
   * `success_vs_collision_rate.png`
   * `optimal_pilots_comparison.png`
   * `intervention_norm_and_jerk.png`

*Nota:* El script genera sus resultados en `./reproduced_results/` para preservar intacto el directorio `results_TFM/`.

### Opción B: cuaderno interactivo Jupyter
Para una inspección interactiva paso a paso con gráficos enriquecidos:
```bash
jupyter notebook notebooks/reproduce_thesis_results.ipynb
```
*(O abrirlo directamente en VSCode, PyCharm o Google Colab y seleccionar **"Run All"**)*.

---

## Nivel 2: inferencia en el simulador Webots (~ 5 - 15 min)

Valida físicamente el comportamiento de los modelos pre-entrenados dentro del simulador Webots con el cuadricóptero Crazyflie.

### Requisitos previos
* Docker y Docker Compose instalados (ver [`docs/docker_guide.md`](docs/docker_guide.md)).
* Modelos pre-entrenados disponibles en `results_TFM/models/`.

### 1. Iniciar el entorno
```bash
# Iniciar contenedor Docker con soporte de Webots
docker compose up -d
docker compose exec webots_rl bash
```

### 2. Evaluar pilotos óptimos (TD3, SAC, PPO)
```bash
# Evaluación de 20 episodios del piloto óptimo determinista:
python3 src/evaluation/metrics.py \
    --backend rltools_td3 \
    --checkpoint results_TFM/models/optimal_pilot_td3_checkpoint_best.h \
    --n-episodes 20 \
    --seed 42 \
    --plot-trajectory-png ./reproduced_results/trayectoria_td3.png
```

### 3. Evaluar pilotos degradados multisemilla (5 semillas: 42, 123, 256, 512, 789)
```bash
bash scripts/evaluate_degraded_pilots.sh
```
Este script recorre automáticamente:
* `NoisyPilot` ($\sigma = 0.5$ y $\sigma = 1.0$)
* `LaggyPilot` ($p_{\text{repeat}} = 0.8$, latencia humana Schaff & Walter)
* `StandByPilot` ($p_{\text{drop}} = 0.8$, pérdida de paquetes con hover failsafe)

### 4. Evaluar el copiloto residual constreñido (placebo vs. asistido)
```bash
bash scripts/evaluate_copilot.sh
```
Calcula cuantitativamente:
* Tasa de Colisión (caídas y salidas de límites).
* Norma media de intervención ($\|a_r\|_2$).
* Métrica de suavidad / Jerk ($\|\Delta a_r\|_2$).

---

## Nivel 3: re-entrenamiento multisemilla desde cero

Para reproducir el entrenamiento completo de las políticas de navegación y asistencia:

### 1. Piloto óptimo TD3 (RLtools C++)
```bash
python3 src/training/train_rltools_td3.py \
    --timesteps 150000 \
    --seed 42 \
    --learning-rate 0.001 \
    --gamma 0.99
```

### 2. Piloto óptimo SAC (RLtools C++)
```bash
python3 src/training/train_rltools_sac.py \
    --timesteps 100000 \
    --seed 42 \
    --target-entropy -4.0
```

### 3. Piloto óptimo PPO (Stable-Baselines3)
```bash
python3 src/training/train_sb3_ppo.py \
    --timesteps 150000 \
    --seed 7 \
    --learning-rate 0.0003 \
    --gamma 0.97
```

### 4. Copiloto residual constreñido (PPO con multiplicador de Lagrange)
```bash
python3 src/training/train_copilot.py \
    --total-timesteps 100000 \
    --warmup-steps 10000 \
    --pilot-type noisy \
    --noise-std 0.50 \
    --init-lambda 1.0 \
    --target-threshold 0.0
```

---

## Organización de datos en `results_TFM/`

El directorio `results_TFM/` contiene los artefactos experimentales que respaldan la memoria:

```
results_TFM/
├── models/                               # Checkpoints pre-entrenados
│   ├── optimal_pilot_td3_checkpoint.h   # Actor TD3 C++ compilado para inferencia estática
│   ├── optimal_pilot_td3_seed_*.h       # Modelos TD3 para semillas 42, 123, 256, 512, 789
│   ├── optimal_pilot_sac_seed_*.h       # Modelos SAC C++
│   ├── ppo_optimal_pilot.zip            # Modelo actor-crítico PPO (Stable-Baselines3)
│   └── copilot_constrained_ppo.zip      # Modelo del copiloto residual PPO
└── logs/                                 # Datos de evaluación y telemetría
    ├── td3/                             # Evaluaciones de 100 episodios y trayectorias PNG
    ├── sac/                             # Evaluaciones SAC y curvas de convergencia
    ├── ppo/                             # Checkpoints intermedios y monitor.csv
    ├── degraded/                        # Evaluaciones multisemilla de pilotos degradados
    └── copilot/                         # Comparativas Placebo vs Asistido
```

---

## Preguntas frecuentes y comportamiento esperado

### ¿Por qué en la evaluación del piloto óptimo la desviación estándar es $0.0$?
La política de inferencia del actor óptimo (TD3, SAC o PPO) es **100% determinista** (se desactiva el ruido de exploración $\sigma_{\text{expl}} = 0$). Dado que el entorno simulado en Webots parte de una posición fija inicial $(0, 0, z_0)$ y la cinemática es determinista, los $100$ episodios siguen la misma trayectoria óptima. Esto es matemáticamente esperado y constituye la prueba de reproducibilidad determinista de la cota superior del sistema. En contraste, los pilotos degradados (ruidosos y lentos) y el sistema asistido incorporan estocasticidad y exhiben varianza controlada dependiente de la semilla.

### ¿Puedo compilar la memoria LaTeX del TFM?
Sí. El código fuente completo del documento se encuentra en la carpeta contigua `UEM_TFM/`. Para compilarlo con referencias bibliográficas:
```bash
cd ../UEM_TFM
pdflatex main.tex
biber main
pdflatex main.tex
```

---

## Licencia y contacto

Este proyecto está bajo licencia **Apache 2.0**.  
Para dudas o consultas respecto a la reproducción de estos experimentos:
* **Autora:** Ana Laura Diedrichs (`ana.diedrichs@docentes.frm.utn.edu.ar`)
* **Repositorio de referencia original (tesis CEIA-FIUBA 2025):** [Zenodo DOI: 10.5281/zenodo.16914761](https://doi.org/10.5281/zenodo.16914761)
