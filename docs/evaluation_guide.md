*Read this in other languages: [Español](evaluation_guide.es.md)*

# Evaluation Guide

Guide for quantitative evaluation of the pilots (optimal, heuristic, noisy, and laggy) in the Webots environment, and computation of statistical metrics analogous to Tables 4.1 through 4.4 of the thesis (Diedrichs, 2025).

---

## Prerequisites

Have the Docker container running with the simulation environment configured (according to [`docker_guide.md`](docker_guide.md) and [`training_guide.md`](training_guide.md)).

---

## 1. Graphics Access (Optional, GUI Mode Only)

If you plan to open the Webots 3D window on your monitor, enable X11 graphics output on your host machine:

```bash
xhost +local:root
```

*(If you are evaluating in **headless mode / without GUI**, you can skip this step entirely).*

---

## 2. Start the Webots Simulator (Choose One Mode)

Choose between visualizing the flight in 3D or evaluating at maximum speed without graphics:

### Mode A: With 3D Graphical Interface (GUI)

Allows visually observing the Crazyflie's flight attitude on your monitor:

```bash
docker exec -it webots_rltools_env bash
webots webots_project/worlds/crazyflie.wbt
```

> **Note:** The Webots graphical window will open and wait for the `<extern>` controller connection.

### Mode B: Headless Mode (No GUI / Ultra-Fast)

Disables the OpenGL engine and runs physics at maximum CPU performance. **20 episodes are evaluated in just 10 to 15 seconds**:

```bash
docker exec -it webots_rltools_env bash
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt
```

**Parameters explained:**
* `xvfb-run -a`: Creates an in-memory virtual display server (*X Virtual Framebuffer*), allowing Webots to run without a physical screen or `xhost` permissions.
* `--batch`: Non-interactive mode (prevents pop-up dialogs or confirmation prompts).
* `--mode=fast`: Computes ODE physics at maximum computational speed (without waiting for the real-time clock).
* `--no-rendering`: **Completely disables 3D drawing and rendering**, saving GPU and CPU cycles.

---

### Alternative: Run Everything in a Single Terminal (Webots in the Background)

If you prefer not to use two separate terminals, you can launch Webots in the background and run the evaluation immediately in the same console:

```bash
# 1. Enter the container
docker exec -it webots_rltools_env bash

# 2. Start headless Webots in the background while silencing logs
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt > /tmp/webots.log 2>&1 &

# 3. Run pilot evaluation (connects within 1 second)
python src/evaluation/metrics.py --pilot-type optimal --checkpoint models/ppo_optimal_pilot.zip --episodes 20 --corner cone_1 --save-csv logs/eval_ppo_optimal.csv

# 4. Once evaluation finishes, stop the background Webots process:
pkill -f webots-bin
```

---

## 3. Terminal 2: Run Pilot Evaluation

Open a second terminal on your machine, enter the container, and run `src/evaluation/metrics.py`.

The script supports evaluating the different pilot types via the `--pilot-type` argument:

### A. Evaluate the Trained Optimal Pilot (PPO)

Evaluates the learned policy **deterministically** (without exploration noise):

```bash
docker exec -it webots_rltools_env bash
python src/evaluation/metrics.py \
  --pilot-type optimal \
  --checkpoint models/ppo_optimal_pilot.zip \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_ppo_optimal.csv
```

### B. Evaluate the Heuristic Pilot (Benchmark / Geometric Reference)

Evaluates the closed-loop proportional controller as a baseline against the optimal pilot:

```bash
python src/evaluation/metrics.py \
  --pilot-type heuristic \
  --heuristic-mode proportional \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_heuristic.csv
```

### C. Evaluate the Noisy Pilot (NoisyPilot - Tremor or Human Error Simulation)

Injects Gaussian perturbations $\mathcal{N}(0, \sigma^2)$ onto the continuous actions of the optimal pilot:

```bash
python src/evaluation/metrics.py \
  --pilot-type noisy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --sigma 0.15 \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_noisy_sigma015.csv
```

### D. Evaluate the Laggy Pilot (LaggyPilot - Reaction Latency Simulation)

Introduces temporal inertia by repeating the previous action with probability $p_{\text{repeat}}$:

```bash
python src/evaluation/metrics.py \
  --pilot-type laggy \
  --checkpoint models/ppo_optimal_pilot.zip \
  --p-repeat 0.25 \
  --episodes 20 \
  --corner cone_1 \
  --save-csv logs/eval_laggy_p025.csv
```

### E. Evaluate the Assisted Pilot (AssistedPilot with Residual Copilot)

Evaluates the additive composition $a = \text{clip}(a_h + a_r, -1, 1)$ by applying the trained copilot on top of a degraded or laggy pilot:

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

## 4. Command-Line Options and Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pilot-type` | `str` | `optimal` | Pilot type: `optimal`, `heuristic`, `noisy`, `laggy`, `assisted`. |
| `--checkpoint` | `str` | `models/ppo_optimal_pilot.zip` | Path to the trained model file (`.zip` or `.h`). |
| `--backend` | `str` | `sb3_ppo` | Backend for OptimalPilot: `sb3_ppo`, `rltools_sac`, `dummy`. |
| `--heuristic-mode` | `str` | `proportional` | Heuristic mode: `proportional` or `open_loop`. |
| `--base-pilot-type` | `str` | `noisy` | Base pilot to assist when `pilot-type=assisted` (`noisy`, `laggy`, `optimal`). |
| `--copilot-checkpoint` | `str` | `None` | Path to the residual copilot model (`.zip` or `.h`). |
| `--copilot-backend` | `str` | `auto` | Copilot backend (`auto`, `sb3_ppo`, `sb3_sac`, `rltools_sac`, `dummy`). |
| `--residual-scale` | `float` | `1.0` | Scaling factor to limit the residual intervention $a_r$. |
| `--episodes` | `int` | `10` | Number of episodes to evaluate. |
| `--corner` | `str` | `cone_1` | Target cone in Webots (`cone_1`, `cone_2`, `cone_3`, `cone_4`). |
| `--max-episode-steps`| `int` | `1000` | Maximum step limit per episode before truncation. |
| `--sigma` | `float` | `0.1` | Standard deviation of noise for `NoisyPilot`. |
| `--p-repeat` | `float` | `0.2` | Action repetition probability for `LaggyPilot`. |
| `--save-csv` | `str` | `logs/eval_results.csv` | Destination path to save detailed per-episode results. |
| `--verbose` | `int` | `1` | `0`: silent, `1`: per-episode and global summary, `2`: step-by-step. |

---

## 5. Metrics Output and CSV Format

### Terminal Output

At the end of the run, a statistical summary is printed:

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

### CSV File (`logs/...csv`)

The generated CSV file contains the following columns per episode:

* `episode`: Sequential episode number.
* `reward`: Cumulative reward obtained in the episode.
* `length`: Number of steps executed before termination.
* `success`: Boolean (`True`/`False`), indicates whether the drone reached the cone at a safe altitude $> 0.20\,\text{m}$.
* `termination`: Termination reason (`goal`, `fall`, `out_of_bounds`, `timeout`).
* `final_distance`: Minimum Euclidean distance in meters to the center of the cone on the final step.

This file can be loaded directly with `pandas` for analysis or plotting in Jupyter notebooks:

```python
import pandas as pd
df = pd.read_csv("logs/eval_ppo_optimal.csv")
print(f"Success rate: {df['success'].mean() * 100:.1f}%")
```

---

## 6. Quantitative Comparison and Report Generation (`compare_pilots.py`)

To systematically aggregate and compare the results of multiple pilots (for instance, contrasting an unassisted pilot against its residual-copilot-assisted counterpart):

```bash
docker exec -it webots_rltools_env bash
python src/evaluation/compare_pilots.py \
  --csv-files \
    logs/eval_heuristic.csv \
    logs/eval_ppo_optimal.csv \
    logs/eval_noisy_sigma015.csv \
    logs/eval_assisted_noisy.csv \
  --labels \
    "Heuristic" \
    "Optimal PPO" \
    "Noisy (unassisted)" \
    "Noisy + Copilot" \
  --output-dir logs/
```

### Generated Artifacts:
1. **Aggregated CSV Table (`logs/pilots_comparison_summary.csv`):** Success rate, collision rate, episode duration, and distances.
2. **Markdown Table (`logs/pilots_comparison_summary.md`):** Formatted for direct inclusion in reports and thesis documents.
3. **PNG Plots (`logs/pilots_success_vs_collision.png` and `logs/pilots_rewards_and_lengths.png`):** High-resolution publication-ready figures.

### 7. Interactive Analysis in Jupyter Notebook

To explore 2D/3D spatial trajectories inside the $2 \times 2\text{ m}$ arena and residual activation curves, open the notebook:
`notebooks/01_pilots_and_copilot_analysis.ipynb`
