*Read this in other languages: [Español](docker_guide.es.md)*

# Docker and Webots Execution Guide

This document describes how to start the project's containerized environment with graphics acceleration (GUI) support and run Webots alongside the reinforcement learning training modules.

---

## 1. Containerized Environment Features

The Docker image defined in [`Dockerfile`](../Dockerfile) and managed via [`docker-compose.yaml`](../docker-compose.yaml) includes:
- **Base OS**: Ubuntu 24.04 LTS.
- **Simulator**: Official Webots (Cyberbotics) installed and configured at `$WEBOTS_HOME=/usr/local/webots`.
- **Graphics support**: X11 server, Mesa OpenGL libraries, and `xvfb` for accelerated or headless simulation.
- **Python environment**: Virtualenv at `/opt/venv` with pre-installed dependencies (`gymnasium`, `stable-baselines3`, `rltools`, `torch`, `tensorboard`, `pandas`, `pybind11`, etc.).
- **Volume mount**: The repository root is mounted at `/workspace`.

---

## 2. Step-by-Step Execution

### Step 1: Allow X11 Display Access from the Host (Linux)
So that Webots can open graphical windows from the container onto your monitor, grant display permissions to the local X server. Run in your host terminal:

```bash
xhost +local:root
# or alternatively:
xhost +local:$USER
```

> [!NOTE]
> This command must be executed once per user session on your host operating system before starting the container if you plan to use the graphical user interface.

---

### Step 2: Configure Hardware Graphics Acceleration (Optional but Recommended)
By default, [`docker-compose.yaml`](../docker-compose.yaml) uses universal software rendering (CPU). For Webots to run at a smooth 60 FPS leveraging your physical machine's GPU, use the local override file:

1. Copy the configuration template:
   ```bash
   cp docker-compose.override.yaml.example docker-compose.override.yaml
   ```
2. Adapt according to your hardware:
   - **Intel / AMD GPU (such as Intel Iris Plus/Xe or Radeon)**:
     Simply keep the `/dev/dri` directive active:
     ```yaml
     services:
       simulador:
         devices:
           - /dev/dri:/dev/dri
     ```
   - **Dedicated NVIDIA GPU**:
     Requires `nvidia-container-toolkit` installed on the host and uncommenting the NVIDIA block in `docker-compose.override.yaml`.
   - **Machines without GPU / Headless servers / CI**:
     Do not create `docker-compose.override.yaml` (or delete it). Docker will use CPU rendering by default without raising any missing-device errors.

> [!TIP]
> The `docker-compose.override.yaml` file is ignored in `.gitignore` so each machine can keep its own hardware acceleration settings without overwriting others.

---

### Step 3: Build and Start the Container
From the repository root (`0001-webots-rl`), build the image (if running for the first time or after changes to `Dockerfile`) and start the service in the background:

```bash
docker compose up -d --build
```

*(If you have already built the image previously and dependencies have not changed, simply run `docker compose up -d`).*

---

### Step 4: Access the Container's Interactive Terminal
To open an interactive Bash session inside the running container:

```bash
docker compose exec -it simulador bash
```
*(Or using the container name: `docker exec -it webots_rltools_env bash`).*

Upon entering:
- You will automatically be in the `/workspace` directory.
- The Python virtual environment (`/opt/venv`) will be active by default in your `$PATH`.

---

### Step 5: Run Webots

Once inside the container terminal, two execution modes are available:

#### Option A: Graphical User Interface Mode (3D GUI)
Ideal for visual inspection, teleoperation, and scenario validation:

```bash
webots /workspace/webots_project/worlds/crazyflie.wbt
```
Or simply launch Webots and open the world from the simulator menu:
```bash
webots
```

#### Option B: Headless Mode (Fast Training)
For large-scale RL training where 3D rendering is not required, you can drastically accelerate simulation steps by running with `xvfb-run` and fast simulation flags:

```bash
xvfb-run webots --batch --mode=fast --no-rendering /workspace/webots_project/worlds/crazyflie.wbt
```

---

### Step 6: Run Training and Tests Inside the Container

From `/workspace` inside the container you can run the scripts directly (see also the [Training Guide](training_guide.md) and the [Evaluation Guide](evaluation_guide.md)):

- **Run unit tests**:
  ```bash
  python -m pytest tests/
  # or individually:
  python tests/test_envs.py
  python tests/test_pilots.py
  ```

- **Train the optimal pilot with PPO (Stable-Baselines3)**:
  ```bash
  python src/training/train_sb3_ppo.py --timesteps 100000
  ```

- **Train the optimal pilot with SAC (RLtools)**:
  ```bash
  python src/training/train_rltools_sac.py --steps 100000
  ```

---

### Step 7: Stop the Container When Finished
When you finish your work session, exit the container terminal with `exit` and stop the services from the host:

```bash
docker compose down
```

---

## 3. Troubleshooting Common Issues

- **`cannot connect to X server` error**:
  Verify that you ran `xhost +local:root` on the host and that the `$DISPLAY` variable is set in your host terminal (`echo $DISPLAY`, usually `:0` or `:1`).
- **Permission issues with files created inside the container**:
  The container shares the local user (`USER=${USER}` in `docker-compose.yaml`). If any generated file ends up owned by `root`, you can fix permissions from the host with:
  ```bash
  sudo chown -R $USER:$USER .
  ```
