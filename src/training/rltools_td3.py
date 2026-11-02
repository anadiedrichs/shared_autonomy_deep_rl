# -*- coding: utf-8 -*-
"""
RLtools-compatible TD3 (Twin Delayed Deep Deterministic Policy Gradient) module.

Implements the TD3 algorithm with the exact same interface and static C/C++
header export structure as RLtools SAC, enabling continuous control training
in Gymnasium/Webots and embedded TinyRL deployment on the STM32F405 microcontroller.
"""

import os
import re
import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


def float_to_c_hex(val: float) -> str:
    """Format a Python float as a C99/C++11 hexfloat literal (e.g. 0X1.4988E4P+0f)."""
    h = float(val).hex()
    return f"{h}f"


_ModuleBase = nn.Module if nn is not None else object


class DeterministicActor(_ModuleBase):
    """
    Deterministic Actor for TD3:
    11-dim continuous observation -> Linear(11, 64) -> ReLU
                                  -> Linear(64, 64) -> ReLU
                                  -> Linear(64, 4)  -> Tanh -> 4-dim action in [-1, 1].
    """

    def __init__(self, obs_dim: int = 11, act_dim: int = 4, hidden_dim: int = 64):
        if nn is None:
            raise ImportError("PyTorch no está disponible. Para entrenar TD3 se requiere 'torch'.")
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


class TwinCritic(_ModuleBase):
    """
    Twin Critic (Q1 and Q2) for TD3:
    Input: [obs, act] (11 + 4 = 15 dims) -> Linear(15, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1).
    """

    def __init__(self, obs_dim: int = 11, act_dim: int = 4, hidden_dim: int = 64):
        if nn is None:
            raise ImportError("PyTorch no está disponible. Para entrenar TD3 se requiere 'torch'.")
        super().__init__()
        in_dim = obs_dim + act_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu)


class ReplayBuffer:
    """Standard experience replay buffer for off-policy RL."""

    def __init__(self, max_size: int = 200_000, obs_dim: int = 11, act_dim: int = 4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32),
            torch.as_tensor(self.act[idx], dtype=torch.float32),
            torch.as_tensor(self.rew[idx], dtype=torch.float32),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32),
            torch.as_tensor(self.done[idx], dtype=torch.float32),
        )


def export_rltools_c_header(actor_or_weights) -> str:
    """
    Exports a trained DeterministicActor (or tuple of weights/biases) as a C/C++ header (.h)
    strictly matching the RLtools namespace, tensor memory, and layer architecture.
    """
    if isinstance(actor_or_weights, (tuple, list)):
        w0, b0, w1, b1, w2, b2 = actor_or_weights
    else:
        # Extract actor weights and biases from PyTorch Module
        w0 = actor_or_weights.net[0].weight.detach().cpu().numpy()  # (64, 11)
        b0 = actor_or_weights.net[0].bias.detach().cpu().numpy()    # (64,)
        w1 = actor_or_weights.net[2].weight.detach().cpu().numpy()  # (64, 64)
        b1 = actor_or_weights.net[2].bias.detach().cpu().numpy()    # (64,)
        w2 = actor_or_weights.net[4].weight.detach().cpu().numpy()  # (4, 64)
        b2 = actor_or_weights.net[4].bias.detach().cpu().numpy()    # (4,)

    # Format arrays as C hexfloats
    def format_arr(arr: np.ndarray) -> str:
        flat = arr.flatten()
        return ", ".join(float_to_c_hex(float(x)) for x in flat)

    mem_w0 = format_arr(w0)
    mem_b0 = format_arr(b0)
    mem_w1 = format_arr(w1)
    mem_b1 = format_arr(b1)
    mem_w2 = format_arr(w2)
    mem_b2 = format_arr(b2)

    header = f"""// NOTE: This code export assumes that the endianness of the target platform is the same as the endianness of the source platform
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/nn/parameters/parameters.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/nn_models/mlp/network.h>

namespace policy {{
    namespace layer_0 {{
        namespace input_layer {{
            namespace weights {{
                namespace parameters_memory {{
                    constexpr float memory[] = {{{mem_w0}}};
                    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 64, 11>;
                    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                    constexpr CONTAINER_TYPE container = {{(float*)memory}}; 
                }}
                using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
                using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Input, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
                constexpr RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
            }}
            namespace biases {{
                namespace parameters_memory {{
                    constexpr float memory[] = {{{mem_b0}}};
                    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 64>;
                    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                    constexpr CONTAINER_TYPE container = {{(float*)memory}}; 
                }}
                using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
                using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Input, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
                constexpr RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
            }}
        }}
        namespace input_layer {{
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 64, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::RELU, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Input>; 
            using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
            using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 11>;
            using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
            using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
            constexpr TYPE module = {{weights::parameters, biases::parameters}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory = {{weights::parameters, biases::parameters}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory_function(){{return T_TYPE{{weights::parameters, biases::parameters}};}}
        }}
        namespace hidden_layer_0 {{
            namespace weights {{
                namespace parameters_memory {{
                    constexpr float memory[] = {{{mem_w1}}};
                    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 64, 64>;
                    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                    constexpr CONTAINER_TYPE container = {{(float*)memory}}; 
                }}
                using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
                using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
                constexpr RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
            }}
            namespace biases {{
                namespace parameters_memory {{
                    constexpr float memory[] = {{{mem_b1}}};
                    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 64>;
                    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                    constexpr CONTAINER_TYPE container = {{(float*)memory}}; 
                }}
                using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
                using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
                constexpr RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
            }}
        }}
        namespace hidden_layer_0 {{
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 64, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::RELU, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Normal>; 
            using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
            using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 64>;
            using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
            using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
            constexpr TYPE module = {{weights::parameters, biases::parameters}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory = {{weights::parameters, biases::parameters}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory_function(){{return T_TYPE{{weights::parameters, biases::parameters}};}}
        }}
        namespace output_layer {{
            namespace weights {{
                namespace parameters_memory {{
                    constexpr float memory[] = {{{mem_w2}}};
                    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 4, 64>;
                    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                    constexpr CONTAINER_TYPE container = {{(float*)memory}}; 
                }}
                using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
                using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Output, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights, true, true>;
                constexpr RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
            }}
            namespace biases {{
                namespace parameters_memory {{
                    constexpr float memory[] = {{{mem_b2}}};
                    using SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 4>;
                    using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Specification<float, unsigned long, SHAPE, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::RowMajorStride<SHAPE>, true>;
                    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::Tensor<SPEC>;
                    constexpr CONTAINER_TYPE container = {{(float*)memory}}; 
                }}
                using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
                using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Specification<TYPE_POLICY, typename parameters_memory::SPEC::TI, typename parameters_memory::SPEC::SHAPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Output, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases, true, true>;
                constexpr RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::Instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
            }}
        }}
        namespace output_layer {{
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Configuration<TYPE_POLICY, unsigned long, 4, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::TANH, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::Output>; 
            using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::BindConfiguration<CONFIG>;
            using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 64>;
            using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
            using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
            constexpr TYPE module = {{weights::parameters, biases::parameters}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory = {{weights::parameters, biases::parameters}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory_function(){{return T_TYPE{{weights::parameters, biases::parameters}};}}
        }}
        namespace model_definition {{
            using TYPE_POLICY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<float>;
            using CONFIG = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::mlp::Configuration<TYPE_POLICY, unsigned long, 4, 1, 64, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::RELU, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::TANH, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::DefaultInitializer<TYPE_POLICY, unsigned long>>;
            using TEMPLATE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::mlp::BindConfiguration<CONFIG>;
            using INPUT_SHAPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<unsigned long, 1, 1, 11>;
            using CAPABILITY = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::capability::Forward<true, true>;
            using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::mlp::NeuralNetwork<CONFIG, CAPABILITY, INPUT_SHAPE>;
            constexpr TYPE module = {{input_layer::factory<TYPE::SPEC::INPUT_LAYER>, {{hidden_layer_0::factory<TYPE::SPEC::HIDDEN_LAYER>}}, output_layer::factory<TYPE::SPEC::OUTPUT_LAYER>}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory = {{input_layer::factory<typename T_TYPE::SPEC::INPUT_LAYER>, {{hidden_layer_0::factory<typename T_TYPE::SPEC::HIDDEN_LAYER>}}, output_layer::factory<typename T_TYPE::SPEC::OUTPUT_LAYER>}};
            template <typename T_TYPE = TYPE>
            constexpr T_TYPE factory_function(){{return T_TYPE{{input_layer::factory_function<typename T_TYPE::SPEC::INPUT_LAYER>(), {{hidden_layer_0::factory_function<typename T_TYPE::SPEC::HIDDEN_LAYER>()}}, output_layer::factory_function<typename T_TYPE::SPEC::OUTPUT_LAYER>()}};}}
        }}
    }}
    using TYPE = layer_0::model_definition::TYPE;
    constexpr TYPE module = layer_0::model_definition::module;
    template <typename T_TYPE = TYPE>
    constexpr T_TYPE factory = layer_0::model_definition::factory<T_TYPE>;
    template <typename T_TYPE = TYPE>
    constexpr T_TYPE factory_function(){{return layer_0::model_definition::factory_function<T_TYPE>();}}
}}
"""
    return header


class RLtoolsParsedPolicy:
    """
    Evaluates an exported RLtools C/C++ header (.h) policy directly in Python using NumPy.
    Supports both TD3 and SAC exported headers.
    """

    def __init__(self, header_content_or_path: str):
        if os.path.exists(header_content_or_path):
            with open(header_content_or_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = header_content_or_path

        self.w0, self.b0, self.w1, self.b1, self.w2, self.b2 = self._parse_weights(content)

    def _parse_weights(self, content: str):
        # Extract all constexpr float memory[] = { ... }; blocks
        matches = re.findall(r"constexpr\s+float\s+memory\[\]\s*=\s*\{([^}]+)\};", content)
        if len(matches) < 6:
            raise ValueError(
                f"No se pudieron encontrar los 6 bloques de memoria de capas en el header. Encontrados: {len(matches)}"
            )

        arrays = []
        for m in matches[:6]:
            tokens = [t.strip().rstrip("fF") for t in m.split(",") if t.strip()]
            nums = [float.fromhex(t) if ("0x" in t.lower() or "0X" in t) else float(t) for t in tokens]
            arrays.append(np.array(nums, dtype=np.float32))

        # Expected dimensions for 11 -> 64 -> 64 -> 4 (or 8 for SAC)
        w0 = arrays[0].reshape(64, 11)
        b0 = arrays[1].reshape(64)
        w1 = arrays[2].reshape(64, 64)
        b1 = arrays[3].reshape(64)

        out_dim = len(arrays[5])
        w2 = arrays[4].reshape(out_dim, 64)
        b2 = arrays[5].reshape(out_dim)

        return w0, b0, w1, b1, w2, b2

    def evaluate(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).flatten()
        # Layer 0: ReLU(w0 @ x + b0)
        h0 = np.maximum(0.0, np.dot(self.w0, x) + self.b0)
        # Layer 1: ReLU(w1 @ h0 + b1)
        h1 = np.maximum(0.0, np.dot(self.w1, h0) + self.b1)
        # Output layer:
        out = np.dot(self.w2, h1) + self.b2

        if len(out) == 4:
            # Deterministic TD3 policy: Tanh squashed
            action = np.tanh(out)
        else:
            # SAC policy: out[:4] is mean, squashed
            action = np.tanh(out[:4])

        return np.clip(action, -1.0, 1.0)

    def action(self, obs: np.ndarray) -> np.ndarray:
        return self.evaluate(obs)


class RLtoolsTD3State:
    """
    Training state matching the RLtools SAC State interface:
    Provides:
        - state.step() -> bool (True when training is complete)
        - state.action(obs) -> np.ndarray (evaluation action)
        - state.export_policy() -> str (C/C++ header content)
    """

    def __init__(
        self,
        env_factory,
        seed: int = 42,
        step_limit: int = 100_000,
        episode_step_limit: int = 1_000,
        gamma: float = 0.97,
        lr: float = 3e-4,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        expl_noise: float = 0.1,
        policy_delay: int = 2,
        batch_size: int = 256,
        warmup_steps: int = 5_000,
        evaluation_interval: int = 1_000,
        num_evaluation_episodes: int = 5,
        verbose: bool = True,
    ):
        self.seed = seed
        self.step_limit = step_limit
        self.episode_step_limit = episode_step_limit
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.eval_interval = evaluation_interval
        self.num_eval_episodes = num_evaluation_episodes
        self.verbose = verbose

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Environment
        self.env = env_factory()
        self.obs_dim = 11
        self.act_dim = 4

        # Models
        self.actor = DeterministicActor(self.obs_dim, self.act_dim, hidden_dim=64)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TwinCritic(self.obs_dim, self.act_dim, hidden_dim=64)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=200_000, obs_dim=self.obs_dim, act_dim=self.act_dim)

        # Tracking state
        self.total_steps = 0
        self.current_obs, _ = self.env.reset(seed=seed)
        self.episode_steps = 0
        self.last_eval_step = 0

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic evaluation action for given observation."""
        self.actor.eval()
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            act = self.actor(t_obs).squeeze(0).cpu().numpy()
        self.actor.train()
        return np.clip(act, -1.0, 1.0)

    def export_policy(self) -> str:
        """Exports policy to RLtools-compatible C/C++ header."""
        return export_rltools_c_header(self.actor)

    def _evaluate_agent(self) -> float:
        """Runs periodic evaluation episodes to log Mean return matching RLtools stdout format."""
        returns = []
        for _ in range(self.num_eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done and ep_len < self.episode_step_limit:
                act = self.action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(act)
                ep_ret += rew
                ep_len += 1
                done = terminated or truncated
            returns.append(ep_ret)

        mean_ret = float(np.mean(returns))
        # Print using the exact format that CStdoutTee intercepts
        print(f"Step: {self.total_steps}/{self.step_limit} Mean return: {mean_ret:.4f}")
        return mean_ret

    def step(self) -> bool:
        """
        Executes one environment and training step.
        Returns:
            bool: True if training has completed (reached step_limit).
        """
        if self.total_steps >= self.step_limit:
            return True

        # 1. Action selection
        if self.total_steps < self.warmup_steps:
            action = np.random.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
        else:
            action = self.action(self.current_obs)
            noise = np.random.normal(0, self.expl_noise, size=self.act_dim).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)

        # 2. Env step
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_steps += 1
        self.total_steps += 1
        done = terminated or truncated

        # 3. Store in replay buffer
        self.replay_buffer.add(self.current_obs, action, reward, next_obs, terminated)

        if done or self.episode_steps >= self.episode_step_limit:
            self.current_obs, _ = self.env.reset()
            self.episode_steps = 0
        else:
            self.current_obs = next_obs

        # 4. Training update
        if self.total_steps >= self.warmup_steps and self.replay_buffer.size >= self.batch_size:
            b_obs, b_act, b_rew, b_next_obs, b_done = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                # Target policy smoothing
                target_noise = torch.randn_like(b_act) * self.policy_noise
                target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
                next_act = torch.clamp(self.actor_target(b_next_obs) + target_noise, -1.0, 1.0)

                # Twin critic target
                q1_target, q2_target = self.critic_target(b_next_obs, next_act)
                q_target = torch.min(q1_target, q2_target)
                y = b_rew + self.gamma * (1.0 - b_done) * q_target

            # Critic loss & update
            q1_curr, q2_curr = self.critic(b_obs, b_act)
            critic_loss = nn.functional.mse_loss(q1_curr, y) + nn.functional.mse_loss(q2_curr, y)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed actor & target network updates
            if self.total_steps % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(b_obs, self.actor(b_obs)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Polyak target update
                for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)
                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                    p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)

        # 5. Periodic evaluation
        if (
            self.total_steps - self.last_eval_step >= self.eval_interval
            and self.total_steps >= self.warmup_steps
        ):
            self.last_eval_step = self.total_steps
            self._evaluate_agent()

        return self.total_steps >= self.step_limit


class TD3:
    """
    RLtools-compatible TD3 class factory.
    Mirrors `rltools.SAC(env_factory, ...)` interface exactly.
    """

    def __init__(
        self,
        env_factory,
        STEP_LIMIT: int = 100_000,
        EPISODE_STEP_LIMIT: int = 1_000,
        GAMMA: float = 0.97,
        evaluation_interval: int = 1_000,
        num_evaluation_episodes: int = 5,
        verbose: bool = True,
        **kwargs,
    ):
        self.env_factory = env_factory
        self.step_limit = STEP_LIMIT
        self.episode_step_limit = EPISODE_STEP_LIMIT
        self.gamma = GAMMA
        self.eval_interval = evaluation_interval
        self.num_eval_episodes = num_evaluation_episodes
        self.verbose = verbose
        self.kwargs = kwargs

    def State(self, seed: int = 42) -> RLtoolsTD3State:
        return RLtoolsTD3State(
            env_factory=self.env_factory,
            seed=seed,
            step_limit=self.step_limit,
            episode_step_limit=self.episode_step_limit,
            gamma=self.gamma,
            evaluation_interval=self.eval_interval,
            num_evaluation_episodes=self.num_eval_episodes,
            verbose=self.verbose,
            **self.kwargs,
        )
