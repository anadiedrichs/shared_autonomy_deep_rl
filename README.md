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

> Status: **initial skeleton**. The modules in `src/` contain the interface and design
> docstrings, pending implementation. See `docs/architecture.md` for details on the
> architectural decisions made prior to porting the code.

## Architectural Decisions (Summary)

- **Stack**: Pure Python. Webots + Gymnasium, without a native C++ controller for now.
- **Optimal pilot**: 100% reinforcement learning. Trained with two algorithms for
  comparison:
  - [`rltools`](https://pypi.org/project/rltools/) (Python wrapper for RLtools) → SAC.
  - [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) → PPO, as a
    "traditional framework" baseline.
  Imitation learning is not used in this iteration (no joystick is available to capture
  high-quality continuous demonstrations; left as future work).
- **Action space**: Continuous, `Box(-1, 1, shape=(4,))` =
  `[vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd]`, reusing the velocity PID controller
  already validated in `drone-deep-rl`.
- **Copilot (shared autonomy)**: Residual policy, following
  [Schaff & Walter, 2020](https://arxiv.org/abs/2004.05097):

  ```
  a = clip(a_h + a_r, -1, 1)
  a_h ~ pi_h(s, g)      # pilot (human or synthetic): goal-directed
  a_r ~ pi_r(s, a_h)    # copilot: residual correction, goal-agnostic
  ```

  The copilot is trained to minimize its own intervention (`‖a_r‖²`) while avoiding
  safety constraint violations (collisions / obstacle proximity).

See the full discussion on why this architecture was chosen (rather than native C++
RLtools or imitation learning) in `docs/architecture.md`.

## Project Structure

```
shared_autonomy_deep_rl/
├── webots_project/            # Webots world (.wbt) and controller
├── src/
│   ├── envs/                  # Gymnasium environments (drone, scenario, pilot+copilot composition)
│   ├── pilots/                # Optimal, noisy, and laggy (synthetic) pilots
│   ├── copilot/               # Residual policy
│   ├── rewards/               # Reward functions (goal-directed / goal-agnostic)
│   ├── training/              # Training scripts (RLtools-SAC, SB3-PPO, copilot)
│   └── evaluation/            # Pilot metrics and comparison
├── notebooks/                 # Exploratory analysis of results
├── tests/
├── docs/architecture.md       # Design decisions and rationale
├── requirements.txt
└── INSTALL.md
```

## Installation

See [`INSTALL.md`](INSTALL.md).

## Technical References

Diedrichs, Ana Laura. «Teleoperación asistida de cuadricópteros mediante aprendizaje por
refuerzo profundo». Zenodo, 2025. <https://doi.org/10.5281/zenodo.16914761>

Schaff, C., & Walter, M. R. (2020). Residual Policy Learning for Shared Autonomy.
*Robotics: Science and Systems (RSS)*. <https://arxiv.org/abs/2004.05097>

Eschmann, J., Albani, D., & Loianno, G. (2024). RLtools: A Fast, Portable Deep
Reinforcement Learning Library for Continuous Control. *JMLR*, 25(301).
<http://jmlr.org/papers/v25/24-0248.html>

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Contact

Ana Diedrichs - ana (dot) diedrichs (at) docentes.frm.utn.edu.ar
