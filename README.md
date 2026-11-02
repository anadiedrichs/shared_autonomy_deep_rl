*Read this in other languages: [Español](README.es.md)*

# Shared Autonomy Deep RL

Integration framework between [RLtools](https://github.com/rl-tools/rl-tools) and the
[Webots](https://cyberbotics.com/) simulator to implement an assisted teleoperation
(shared autonomy) system for Crazyflie quadrotors via residual policy learning in a
continuous action space.

Continuation of the thesis work by Ana Laura Diedrichs, *"Teleoperación asistida de
cuadricópteros mediante aprendizaje por refuerzo profundo"* (*"Assisted teleoperation of
quadrotors via deep reinforcement learning"*, CEIA-FIUBA, 2025,
[Zenodo](https://doi.org/10.5281/zenodo.16914761)), and the
[drone-deep-rl](https://github.com/anadiedrichs/drone-deep-rl) repository, within the
framework of the research project *"Shared Autonomy for UAVs: Sim-to-Real Development with
Webots and RLtools"* (Universidad Europea).

> **Project Status**: **Master's Thesis Complete Framework**. Fully implemented continuous navigation policies (RLtools TD3/SAC and SB3 PPO), synthetic degraded human pilot models (NoisyPilot, LaggyPilot, StandByPilot), and the Constrained Residual Copilot with adaptive Lagrange multipliers (Schaff & Walter, 2020). Experimental models, evaluations, and figures are organized under `results_TFM/`.


## 🔬 Quick Reproducibility (For Evaluators)

To immediately reproduce all tables (Tables 4.1 to 4.5) and figures from the Master's Thesis without installing Webots or running heavy training:

```bash
# Level 1: Offline table and figure generation (< 1 min, zero-dependency)
python3 scripts/reproduce_results.py
```

Or run the interactive notebook:
* [`notebooks/reproduce_thesis_results.ipynb`](notebooks/reproduce_thesis_results.ipynb)

For the complete 3-tier reproducibility protocol (including Webots simulation evaluation and re-training from scratch), consult the master document:
👉 [**REPRODUCIBILITY.md**](REPRODUCIBILITY.md) (or [Guía en Español](REPRODUCIBILIDAD.md)).


## Architectural Decisions (Summary)

- **Stack**: Pure Python + C++ backend. Webots + Gymnasium, leveraging the high-speed C++ engine of [RLtools](https://github.com/rl-tools/rl-tools) for zero-dependency static header inference alongside PyTorch / Stable-Baselines3.
- **Optimal pilot**: 100% reinforcement learning. Evaluated across three continuous actor-critic algorithms:
  - [`rltools`](https://pypi.org/project/rltools/) → TD3 & SAC (C++ embedded deployment format `.h`).
  - [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) → PPO baseline (`.zip`).
- **Action space**: Continuous, `Box(-1, 1, shape=(4,))` =
  `[vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd]`, reusing the velocity PID controller
  validated in `drone-deep-rl`.
- **Copilot (shared autonomy)**: Constrained Residual Policy, following
  [Schaff & Walter, 2020](https://arxiv.org/abs/2004.05097):

  ```
  a = clip(a_h + beta * a_r, -1, 1)
  a_h ~ pi_h(s, g)      # pilot (human or synthetic): goal-directed
  a_r ~ pi_r(s, a_h)    # copilot: residual correction, goal-agnostic safety barrier
  ```

  The copilot minimizes its own intervention norm (`‖a_r‖²`) while enforcing safety constraints (obstacle proximity, wall boundaries, ground collision avoidance) updated dynamically via a Lagrange multiplier.

See the full architectural discussion in [`docs/architecture.md`](docs/architecture.md).

---

## Project Structure

```
0001-webots-rl/
├── REPRODUCIBILITY.md         # Master guide for reproducing thesis results (3 tiers)
├── REPRODUCIBILIDAD.md        # Guía de reproducibilidad en español
├── webots_project/            # Webots simulation world (.wbt) and supervisor controller
├── results_TFM/               # Experimental models (.h, .zip) and CSV evaluations
│   ├── models/                # Trained checkpoints (TD3, SAC, PPO, Copilot)
│   └── logs/                  # Multi-seed 100-episode logs and trajectories
├── scripts/                   # Automated training and evaluation runners
│   ├── reproduce_results.py   # Standalone table and figure reproduction script
│   ├── train_all_seeds.sh     # Multi-seed training pipeline
│   ├── evaluate_degraded_pilots.sh # Degraded pilot evaluation suite
│   ├── train_copilot.sh       # Constrained PPO copilot training runner
│   └── evaluate_copilot.sh    # Copilot Placebo vs. Assisted evaluation runner
├── notebooks/                 # Interactive Jupyter notebooks for analysis
│   ├── reproduce_thesis_results.ipynb  # Thesis results reproduction notebook
│   └── 01_pilots_and_copilot_analysis.ipynb
├── src/
│   ├── envs/                  # Continuous Gymnasium environments & SharedAutonomyEnv
│   ├── pilots/                # BasePilot, OptimalPilot, NoisyPilot, LaggyPilot, StandByPilot
│   ├── copilot/               # ConstrainedPPO (Lagrange multiplier residual policy)
│   ├── rewards/               # Goal-directed and general safety reward formulations
│   ├── training/              # RL training workflows (TD3, SAC, SB3 PPO, Copilot)
│   └── evaluation/            # Quantitative metrics, trajectory plotting, comparisons
├── docs/                      # Architectural, Docker, training, and evaluation guides
└── INSTALL.md                 # Environment setup and installation instructions
```

---

## Installation & Simulator Execution

See [`INSTALL.md`](INSTALL.md) and [`docs/docker_guide.md`](docs/docker_guide.md).

---

## Technical References

Diedrichs, Ana Laura. «Teleoperación asistida de cuadricópteros mediante aprendizaje por
refuerzo profundo». Zenodo, 2025. <https://doi.org/10.5281/zenodo.16914761>

Schaff, C., & Walter, M. R. (2020). Residual Policy Learning for Shared Autonomy.
*Robotics: Science and Systems (RSS)*. <https://arxiv.org/abs/2004.05097>

Eschmann, J., Albani, D., & Loianno, G. (2024). RLtools: A Fast, Portable Deep
Reinforcement Learning Library for Continuous Control. *JMLR*, 25(301).
<http://jmlr.org/papers/v25/24-0248.html>

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Contact

Ana Diedrichs - ana (dot) diedrichs (at) docentes.frm.utn.edu.ar
