"""
model.py
--------
Factory for building the PPO agent. Keeping model construction in its own
module (separate from training logic) is best practice: it makes the model
reusable in `evaluate.py`, easier to unit-test, and keeps `train.py` focused
purely on the training loop.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

import config


def build_ppo_model(env: VecEnv) -> PPO:
    """Instantiate a PPO model with the project's hyperparameters.

    The `MlpPolicy` is used because the default highway-v0 observation is
    a small kinematic matrix (5 × 5 floats), not pixels — so a simple
    fully-connected network is both sufficient and CPU-friendly.

    Default SB3 architecture for `MlpPolicy`:
        - Two hidden layers, 64 units each
        - Tanh activation
        - Separate policy and value-function heads

    Args:
        env: A vectorised gymnasium environment.

    Returns:
        A configured (but untrained) PPO instance.
    """
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        learning_rate=config.LEARNING_RATE,
        clip_range=config.CLIP_RANGE,
        tensorboard_log=config.TENSORBOARD_LOG_DIR,
        verbose=1,
    )
    return model


def load_ppo_model(checkpoint_path: str, env: VecEnv | None = None) -> PPO:
    """Load a previously saved PPO checkpoint.

    Args:
        checkpoint_path: Path to the saved .zip model (without extension).
        env:             Optional environment to attach to the loaded model.

    Returns:
        The reconstructed PPO instance.
    """
    return PPO.load(checkpoint_path, env=env)