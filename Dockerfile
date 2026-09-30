# Use Ubuntu 24.04 LTS as clean base
FROM ubuntu:24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies, Python, C++, and graphics tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    git \
    build-essential \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    pybind11-dev \
    mesa-utils \
    libgl1 \
    libglx-mesa0 \
    libxcb-cursor0 \
    libblas-dev \
    locales \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 2. Add Cyberbotics official repository and install Webots
RUN install -m 0755 -d /etc/apt/keyrings && \
    wget -qO- https://cyberbotics.com/Cyberbotics.asc | tee /etc/apt/keyrings/Cyberbotics.asc > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/Cyberbotics.asc] https://cyberbotics.com/debian/ binary-amd64/" | tee /etc/apt/sources.list.d/cyberbotics.list && \
    apt-get update && \
    apt-get install -y webots && \
    rm -rf /var/lib/apt/lists/*

# Configure Webots environment variables
ENV WEBOTS_HOME=/usr/local/webots
ENV PATH="${WEBOTS_HOME}:${PATH}"
ENV QTWEBENGINE_DISABLE_SANDBOX=1

# 3. Create and activate a Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# 4. Install Python packages for reinforcement learning
RUN pip install --upgrade pip && \
    pip install numpy pybind11 h5py tensorboard rltools gymnasium stable-baselines3 pandas rich tqdm

# Working directory setup
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
