# -*- coding: utf-8 -*-
"""
Unit tests for RLtools-compatible TD3 algorithm, C header export, and inference.
"""

import os
import sys
import tempfile
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.training.rltools_td3 import (
    DeterministicActor,
    export_rltools_c_header,
    RLtoolsParsedPolicy,
    TD3,
)
from src.pilots.optimal_pilot import OptimalPilot


def test_rltools_td3_c_header_export_and_bit_exact_parsing():
    """Validates that C header export and NumPy bit-exact parser produce identical actions."""
    np.random.seed(42)
    w0 = np.random.randn(64, 11).astype(np.float32) * 0.1
    b0 = np.random.randn(64).astype(np.float32) * 0.05
    w1 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b1 = np.random.randn(64).astype(np.float32) * 0.05
    w2 = np.random.randn(4, 64).astype(np.float32) * 0.1
    b2 = np.random.randn(4).astype(np.float32) * 0.05

    sample_obs = np.array(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.0],
        dtype=np.float32,
    )
    # Manual forward pass
    h0 = np.maximum(0.0, np.dot(w0, sample_obs) + b0)
    h1 = np.maximum(0.0, np.dot(w1, h0) + b1)
    expected_act = np.tanh(np.dot(w2, h1) + b2)

    # Export C header from weights
    c_header = export_rltools_c_header((w0, b0, w1, b1, w2, b2))
    assert "namespace policy" in c_header
    assert "constexpr float memory[]" in c_header
    assert "rl_tools" in c_header

    # Parse with standalone NumPy loader
    parsed = RLtoolsParsedPolicy(c_header)
    parsed_act = parsed.evaluate(sample_obs)

    # Validate numerical equivalence
    np.testing.assert_allclose(parsed_act, expected_act, atol=1e-5)
    assert parsed_act.shape == (4,)
    assert np.all(parsed_act >= -1.0) and np.all(parsed_act <= 1.0)

    # If torch is available, also test via DeterministicActor
    if torch is not None:
        actor = DeterministicActor(obs_dim=11, act_dim=4, hidden_dim=64)
        actor.net[0].weight.data.copy_(torch.from_numpy(w0))
        actor.net[0].bias.data.copy_(torch.from_numpy(b0))
        actor.net[2].weight.data.copy_(torch.from_numpy(w1))
        actor.net[2].bias.data.copy_(torch.from_numpy(b1))
        actor.net[4].weight.data.copy_(torch.from_numpy(w2))
        actor.net[4].bias.data.copy_(torch.from_numpy(b2))
        actor.eval()

        with torch.no_grad():
            torch_act = actor(torch.as_tensor(sample_obs).unsqueeze(0)).squeeze(0).numpy()
        np.testing.assert_allclose(parsed_act, torch_act, atol=1e-5)


def test_rltools_td3_training_state_mock():
    """Validates that TD3 training state steps, updates, and exports policy in mock environment."""
    if torch is None:
        print("[Test] PyTorch no instalado en este entorno host; omitiendo test_rltools_td3_training_state_mock.")
        return

    from src.training.train_rltools_sac import make_env_factory

    factory = make_env_factory(corner_name="cone_1", max_episode_steps=50)
    td3 = TD3(
        factory,
        STEP_LIMIT=15,
        EPISODE_STEP_LIMIT=50,
        warmup_steps=5,
        batch_size=4,
        evaluation_interval=10,
        num_evaluation_episodes=2,
        verbose=False,
    )
    state = td3.State(seed=42)

    # Step through warmup and training
    finished = False
    for _ in range(15):
        finished = state.step()

    assert finished is True
    assert state.total_steps >= 15

    # Export policy
    exported_h = state.export_policy()
    assert isinstance(exported_h, str)
    assert len(exported_h) > 500


def test_optimal_pilot_rltools_td3_integration():
    """Validates that OptimalPilot loads exported TD3 header and predicts valid actions."""
    np.random.seed(42)
    w0 = np.random.randn(64, 11).astype(np.float32) * 0.1
    b0 = np.zeros(64, dtype=np.float32)
    w1 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b1 = np.zeros(64, dtype=np.float32)
    w2 = np.random.randn(4, 64).astype(np.float32) * 0.1
    b2 = np.zeros(4, dtype=np.float32)

    c_header = export_rltools_c_header((w0, b0, w1, b1, w2, b2))

    with tempfile.NamedTemporaryFile(suffix=".h", mode="w", delete=False) as f:
        f.write(c_header)
        temp_header_path = f.name

    try:
        pilot = OptimalPilot(backend="rltools_td3", checkpoint_path=temp_header_path)
        obs = np.random.uniform(-1.0, 1.0, size=11).astype(np.float32)
        act = pilot.get_action(obs)

        assert isinstance(act, np.ndarray)
        assert act.shape == (4,)
        assert np.all(act >= -1.0) and np.all(act <= 1.0)

        # Alias choose_action
        act2, aux = pilot.choose_action(obs)
        np.testing.assert_array_equal(act, act2)
        assert aux == []
    finally:
        if os.path.exists(temp_header_path):
            os.remove(temp_header_path)
