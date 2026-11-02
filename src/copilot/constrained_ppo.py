# -*- coding: utf-8 -*-
"""
Constrained Proximal Policy Optimization (ConstrainedPPO) with Lagrange Multiplier.

Implements the constrained policy optimization framework from Schaff & Walter (2020)
[Residual Policy Learning for Shared Autonomy, RSS 2020] on top of Stable-Baselines3 PPO:

1. Combined Action:
    a_total = a_h + beta * a_r
   where a_h is proposed by the surrogate pilot, a_r is the bounded residual correction,
   and beta is the authority scaling factor.

2. Constrained Objective:
    min_pi E[ ||a_r||_2^2 ]  s.t.  E[ R_general ] >= target_d

3. Dual Lagrange Formulation:
    L_actor = (1 / (1 + lambda_val)) * ||a_r||_2^2 + (lambda_val / (1 + lambda_val)) * L_PPO
    where lambda_val = softplus(log_lambda) > 0 strictly ensures non-negativity.

4. Dual Multiplier Update:
    L_lambda = -softplus(log_lambda) * (target_d - mean_R_general)
    Updated via dual gradient ascent: when performance is below target_d, lambda increases
    prioritizing task performance/safety; when performance satisfies target_d, lambda decreases
    driving residual intervention toward zero (minimal intervention principle).

5. TensorFlow / TensorBoard Logging:
    Custom metrics are logged via SB3's logger to TensorBoard (.tfevents format):
    - train/lambda
    - train/loss_lambda
    - train/mean_residual_norm
    - train/ppo_loss
    - train/r_general_mean
"""

from typing import Any, ClassVar, TypeVar
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import explained_variance
except ImportError as err:
    raise ImportError(
        f"Stable-Baselines3 is required to use ConstrainedPPO: {err}. "
        "Please install it via 'pip install stable-baselines3'."
    )

SelfConstrainedPPO = TypeVar("SelfConstrainedPPO", bound="ConstrainedPPO")


class ConstrainedPPO(PPO):
    """
    Constrained PPO algorithm with adaptive Lagrange multiplier for Residual Shared Autonomy.
    """

    def __init__(
        self,
        *args,
        target_threshold: float = 0.0,
        init_lambda: float = 1.0,
        lr_lambda: float = 3e-4,
        residual_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initializes ConstrainedPPO.

        Args:
            target_threshold: Minimum expected return threshold (target_d) for task/safety constraint.
            init_lambda: Initial value for the Lagrange multiplier lambda (default: 1.0).
            lr_lambda: Learning rate for the dual Lagrange multiplier optimizer (default: 3e-4).
            residual_weight: Scaling coefficient for the residual norm penalty.
            *args, **kwargs: Standard arguments passed to stable_baselines3.PPO.
        """
        super().__init__(*args, **kwargs)

        self.target_threshold = float(target_threshold)
        self.init_lambda = float(init_lambda)
        self.lr_lambda = float(lr_lambda)
        self.residual_weight = float(residual_weight)

        # Inverse softplus initialization: init_lambda = softplus(log_lambda)
        # softplus(x) = ln(1 + e^x) -> x = ln(e^init_lambda - 1)
        if self.init_lambda > 0.0:
            inv_sp = float(np.log(np.exp(min(self.init_lambda, 20.0)) - 1.0 + 1e-6))
        else:
            inv_sp = -5.0

        # Declare log_lambda parameter on the correct PyTorch device
        self.log_lambda = nn.Parameter(
            th.tensor([inv_sp], dtype=th.float32, device=self.device)
        )

        # Independent Adam optimizer for dual variable
        self.lambda_optimizer = th.optim.Adam([self.log_lambda], lr=self.lr_lambda)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Registers PyTorch parameters to be serialized in model.save() / model.load().
        """
        state_dicts, saved_pytorch_variables = super()._get_torch_save_params()
        saved_pytorch_variables = list(saved_pytorch_variables) + ["log_lambda"]
        return state_dicts, saved_pytorch_variables

    @property
    def lambda_val(self) -> th.Tensor:
        """Computes positive Lagrange multiplier lambda = softplus(log_lambda)."""
        return F.softplus(self.log_lambda)

    def train(self) -> None:
        """
        Executes one update step over the collected rollout buffer with constrained loss.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        residual_losses = []
        clip_fractions = []

        # Mean empirical return from rollout buffer (R_general)
        mean_r_general = self.rollout_buffer.returns.mean()

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                # Re-evaluate actions with current policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate PPO loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Residual action norm penalty ||a_r||_2^2 (Schaff & Walter, 2020)
                # In SharedAutonomyEnv, the actions executed by this policy are the residual corrections a_r
                residual_norm_sq = th.sum(actions**2, dim=-1).mean()
                residual_losses.append(residual_norm_sq.item())

                # Composite Lagrangian Actor Loss:
                # L_actor = (1 / (1 + lambda)) * ||a_r||_2^2 + (lambda / (1 + lambda)) * L_PPO
                current_lambda = self.lambda_val.detach()
                weight_residual = (1.0 / (1.0 + current_lambda)) * self.residual_weight
                weight_ppo = current_lambda / (1.0 + current_lambda)

                actor_loss = weight_residual * residual_norm_sq + weight_ppo * policy_loss

                # Value loss
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Total loss for actor-critic update
                loss = actor_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step for Actor-Critic
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

        # ----------------------------------------------------------------------
        # Dual Step: Update Lagrange Multiplier lambda
        # ----------------------------------------------------------------------
        # L_lambda = -softplus(log_lambda) * (target_d - mean_R_general)
        # Minimizing L_lambda performs gradient ascent on lambda:
        # If mean_R_general < target_threshold, constraint is violated -> lambda increases
        # If mean_R_general >= target_threshold, constraint is satisfied -> lambda decreases
        constraint_gap = self.target_threshold - mean_r_general
        loss_lambda = -F.softplus(self.log_lambda) * constraint_gap

        self.lambda_optimizer.zero_grad()
        loss_lambda.backward()
        self.lambda_optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # ----------------------------------------------------------------------
        # TensorBoard / TensorFlow Logging (.tfevents format)
        # ----------------------------------------------------------------------
        lambda_num = float(self.lambda_val.item())
        self.logger.record("train/lambda", lambda_num)
        self.logger.record("train/loss_lambda", float(loss_lambda.item()))
        self.logger.record("train/mean_residual_norm", float(np.mean(residual_losses)))
        self.logger.record("train/r_general_mean", float(mean_r_general.item()))
        self.logger.record("train/target_threshold", float(self.target_threshold))
        self.logger.record("train/constraint_gap", float(constraint_gap.item()))
        self.logger.record("train/ppo_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
