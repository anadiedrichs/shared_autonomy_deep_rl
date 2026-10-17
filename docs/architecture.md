*Read this in other languages: [Español](architecture.es.md)*

# Architecture Decisions

This document records the architectural decisions made prior to porting code from
[`drone-deep-rl`](https://github.com/anadiedrichs/drone-deep-rl) (CEIA-FIUBA 2025 thesis) to
this repository, along with the rationale behind each one. The goal is for any
collaborator (or the author herself, months later) to understand *why* alternative
options were discarded, not just which one was chosen.

## 1. RLtools–Webots Integration: Python Wrapper (not Native C++)

RLtools is a **pure C++ header-only** library with zero dependencies, designed so that
the compiler knows the dimensions of all data structures (observations, actions, network
layers) at compile time. The RLtools team also distributes a Python wrapper
(`pip install rltools`, repo
[`rl-tools/python-interface`](https://github.com/rl-tools/python-interface)) that:

- Depends only on `pybind11`.
- Takes as input an `env_factory()` function returning a **Gymnasium** environment, and
  infers the observation/action dimensions from it to compile at runtime a "bridge"
  environment compatible with RLtools.
- Currently **only exposes the SAC training loop**, with limited hyperparameter control
  ("work in progress").
- Allows exporting the trained checkpoint as a **C header** (`state.export_policy()`),
  intended for direct deployment on embedded firmware — the piece that connects to the
  portability / sim-to-real objective of the project, even without writing the Webots
  controller in C++.

### Options Considered

| Option | Description | Why it was (or was not) chosen |
|---|---|---|
| A | Pure C++ Webots controller + native RLtools | Maximum performance and fidelity to the TinyRL goal, but requires rewriting all of `DroneRobotSupervisor`, sensors, PID, and pilots in C++ from scratch. Higher schedule risk. Kept as a future evolution once the architecture is validated in Python. |
| B | RLtools Python wrapper | **Chosen.** Reuses `DroneRobotSupervisor` in Python almost as-is. Known limitation: SAC only for now. |
| C | Hybrid (SB3 in Python + separate C++ module with RLtools) | Discarded due to duplicated maintenance; a single Python codebase (Option B) plus SB3 as a baseline within the same stack was preferred. |

### Consequence for the Goal "Compare PPO and SAC Natively Integrated with RLtools"

The RLtools Python wrapper does not support PPO training. Therefore, the PPO vs. SAC
comparison is resolved as follows:

- **SAC via `rltools`**: candidate for the "optimized/portable model" (a policy that can
  be exported to C and potentially deployed on the real Crazyflie).
- **PPO via `stable-baselines3`**: "traditional framework" baseline, which also directly
  matches the project's evaluation objective ("compare... against baseline pilots and
  models trained in traditional frameworks").

This is explicitly documented so that in the final report/paper it is clear that the
asymmetry (SAC-RLtools vs. PPO-SB3, instead of both PPO/SAC in RLtools) is a known
limitation of the library in its current version, not a design oversight.

## 2. Optimal Pilot: Pure RL, Without Imitation Learning (For Now)

In the 2025 thesis, the optimal pilot was obtained using two approaches: RL (PPO/DQN/A2C
with discrete actions) and imitation learning (Behavioral Cloning with the `imitation`
library, from keyboard-captured demonstrations).

When moving to **continuous** actions, capturing demonstrations via keyboard is no longer
straightforward: without a joystick, only "bang-bang" commands can be captured (maximum
value while holding a key, zero when released), which fail to exploit the continuous
range of the action space.

### Options Considered for the Optimal Pilot

1. **Pure RL (chosen)**: RLtools-SAC + SB3-PPO, without human capture. It aligns with the
   primary specific objective of the project (training and comparing PPO/SAC) and does
   not depend on acquiring hardware.
2. Keyboard "bang-bang" capture mapped to the extremes of `Box(-1,1)`: discarded for now;
   kept as a potential BC fallback if RL fails to converge well in some scenario (as
   happened in the 2025 thesis with pure PPO).
3. Virtual joystick (mouse/trackpad → continuous axes): listed as an **optional** task in
   the project proposal ("Develop the control infrastructure..."). Not implemented in
   this first iteration.
4. Physical USB gamepad/joystick: requires hardware acquisition, out of immediate scope.

## 3. Continuous Action Space

```
Box(low=-1, high=1, shape=(4,), dtype=float32)
# [vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd] (normalized)
```

The velocity PID controller already implemented in `DroneRobotSupervisor` (2025 thesis)
is reused: the original 6 discrete actions were fixed combinations of these 4 values, so
transitioning to continuous control does not require redesigning the low-level controller—
only denormalizing the action vector before passing it to the PID (the same pattern used
by the RLtools example itself with `gymnasium.wrappers.RescaleAction`).

## 4. Shared Autonomy: Residual Policy

Following [Schaff & Walter (2020)](https://arxiv.org/abs/2004.05097):

```
a = a_h + a_r
a_h ~ pi_h(s, g)      # pilot: human or synthetic (optimal / noisy / laggy)
a_r ~ pi_r(s, a_h)    # copilot: residual correction, conditioned on state and pilot action
```

This replaces the discrete scheme from the 2025 thesis (distance table `Q(a*, ah)` +
threshold `alpha`, which selected *between* the pilot's action and the copilot's action)
with an *additive* and continuous combination, where the copilot:

- Is trained against a **goal-agnostic** reward (`reward_general_safety.py`): penalizes
  collision/proximity to obstacles and penalizes the magnitude `‖a_r‖²`, so that it
  intervenes as little as possible.
- Does not know the pilot's goal (unlike the pilot, which is trained with a
  goal-directed reward, `reward_goal_directed.py`).

## 5. Webots Execution Model: External Controller (`<extern>`)

In the 2025 thesis (`drone-deep-rl`), the proven workflow consisted of opening Webots and
**running the control and training scripts directly from the IDE (Visual Studio Code)**
or the terminal.

To preserve this workflow and decouple the Gymnasium environment from the Webots GUI, the
Crazyflie node in the `.wbt` world (`webots_project/worlds/crazyflie.wbt`) is configured
with:

```vrml
DEF crazyflie Crazyflie {
  ...
  controller "<extern>"
  supervisor TRUE
}
```

### Technical Justification:

1. **Inversion of Control Required by Deep RL**: In frameworks such as Stable-Baselines3
   (PPO) and RLtools (SAC), the main control loop belongs to the learning algorithm
   (`model.learn()` and `env.step()`), not the simulator. By operating as an external
   controller, the Python process acts as the master process that steps the physics
   simulation via IPC (`supervisor.step(timestep)`), allowing the simulator to pause
   during GPU gradient computation and optimization updates.
2. **Execution Flexibility (IDE / Terminal / Docker)**: Allows running evaluation scripts
   ([`src/evaluation/run_heuristic_episode.py`](../src/evaluation/run_heuristic_episode.py))
   or training scripts from any terminal, VS Code, or Docker container
   (`webots_rltools_env`), automatically connecting to the active physics simulation via
   shared memory (`ipc: host`).
3. **Accelerated Headless Simulation**: Enables training at speeds far exceeding real
   time (10x–50x) via Webots headless flags:
   `webots --batch --mode=fast --no-rendering <world.wbt>`.

## Open Questions / Pending Confirmation

- Exact parameterization of `reward_general_safety.py`: relative weights between collision
  penalty and `‖a_r‖²` intervention penalty.
- Whether the residual policy is trained with RLtools-SAC (consistent with the rest of the
  pipeline) or with SB3, and whether it is retrained for each pilot type
  (optimal/noisy/laggy) or trained against a single fixed pilot and evaluated across the
  others.
- How to export/visualize RLtools training curves in TensorBoard (SB3 supports it
  natively; RLtools has its own logging mechanism to review).
