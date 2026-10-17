*Read this in other languages: [Español](training_guide.es.md)*

# Training Guide

How-to guide for training and visualizing reinforcement learning training in Webots.

In this guide we show how to train an optimal pilot with PPO, with SAC (RLtools), and the residual copilot.

## Prerequisite

Have the environment configured as described in [`docker_guide.md`](docker_guide.md) in this repository.

## 1. Allow Access to the Graphical Interface (in Your Local Host Terminal)

If you just logged in or opened a new terminal session on your system, make sure to enable X11 display access for Docker:

```bash
xhost +local:root
```

## 2. Terminal 1: Open the Webots Simulator with GUI

Enter the container and open the `crazyflie.wbt` world:

```bash
docker exec -it webots_rltools_env bash
webots webots_project/worlds/crazyflie.wbt
```

> The Webots graphical window will open. Since the Crazyflie has its controller configured in `<extern>` mode, the simulator will wait for the training script to connect.

## 3. Terminal 2: Launch Training

Choose between training with **PPO (Stable-Baselines3)**, **SAC (RLtools in C++)**, or training the **Residual Copilot**:

### Option A: Training with PPO (Stable-Baselines3)

Open a second terminal on your machine, enter the container, and run `train_sb3_ppo.py`:

```bash
docker exec -it webots_rltools_env bash
python src/training/train_sb3_ppo.py --timesteps 100000 --max-episode-steps 1000 --corner cone_1
```

* Saves the model to `models/ppo_optimal_pilot.zip` and writes logs to `logs/ppo`.
* You can adjust `--timesteps` (e.g., `20000` for a quick test or `100000` for convergence).

### Option B: Training with SAC (RLtools) and Exporting to a C Header

Run the Soft Actor-Critic algorithm implemented in C++ via RLtools:

```bash
docker exec -it webots_rltools_env bash
python src/training/train_rltools_sac.py \
  --steps 100000 \
  --max-episode-steps 1000 \
  --corner cone_1 \
  --export-path models/optimal_pilot_sac_checkpoint.h \
  --log-dir logs/sac
```

* **Just-in-time C++ compilation:** RLtools automatically compiles the C++ interface using `g++` and `BLAS`.
* **C Header export:** Upon completing the training steps, it exports the trained policy directly as a C header file (`models/optimal_pilot_sac_checkpoint.h`), ready for ultra-lightweight embedded inference on the Crazyflie microcontroller.

### Option C: Training the Residual Copilot (Shared Autonomy)

Train the residual policy $\pi_r(s, a_h)$ while keeping a human or degraded synthetic pilot (`noisy` or `laggy`) frozen, optimizing the *goal-agnostic* safety reward (Schaff & Walter, 2020):

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

* **Key parameters:**
  - `--algorithm`: RL algorithm (`ppo` or `sac`).
  - `--pilot-type`: Frozen pilot to assist (`noisy`, `laggy`, `heuristic`, `optimal`).
  - `--lambda-intervention`: Magnitude penalty $\|a_r\|^2$ (default `0.10`); higher values restrict copilot intervention more strongly.
  - `--residual-scale`: Attenuation factor limiting the copilot's authority over the pilot's commands (in $(0, 1]$).

---

## 4. How to Control Visualization Inside Webots

Once the script connects, the simulation will start automatically:

* **Simulation speed (Webots top toolbar):**
  - **Play (Real-time):** Allows observing quadrotor physics, pitch/roll attitude, and altitude at natural speed.
  - **Fast / Run:** Accelerates computational simulation to the maximum speed of your CPU/GPU. You will see the drone move in fast-forward from one attempt to the next, enabling faster training while still viewing the 3D animation.
* **What to expect during episodes:**
  - On each reset, the drone takes off and stabilizes at 0.50 m.
  - During early episodes, it will explore with noisier movements; if it collides with the walls or crashes, the episode resets immediately.
  - As training progresses, it will consolidate direct trajectories toward `cone_1` (+0.9, +0.9).

---

## 5. Fast Training in Headless Mode (No GUI)

If you prefer to train at maximum computational speed without spending GPU cycles or rendering 3D windows:

```bash
docker exec -it webots_rltools_env bash
xvfb-run -a webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt
```
Then, in Terminal 2, run the chosen training script (PPO or SAC).

---

## 6. (Optional) View Metrics in TensorBoard

To monitor reward curves, episode lengths, and algorithm losses in your browser:

```bash
# To monitor PPO:
tensorboard --logdir ./logs/ppo

# To monitor SAC (RLtools):
tensorboard --logdir ./logs/sac

# To monitor the Residual Copilot:
tensorboard --logdir ./logs/copilot

# Or to compare all training runs at once:
tensorboard --logdir ./logs
```

Then open in your browser: [http://localhost:6006](http://localhost:6006). To evaluate the trained models, see [`evaluation_guide.md`](evaluation_guide.md).
