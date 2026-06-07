"""
utils.py
--------
Shared utility functions: environment creation, directory setup,
and reward-curve plotting.
"""

import os
import gymnasium as gym
import highway_env  # noqa: F401  — registers highway-env envs with gymnasium
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config


def make_env(env_id: str = config.ENV_ID) -> gym.Env:
    """Create and wrap a single Highway-Env gymnasium environment.

    The Monitor wrapper records episode rewards and lengths, which
    Stable-Baselines3 uses internally for logging.

    Args:
        env_id: Gymnasium environment ID string.

    Returns:
        A Monitor-wrapped gymnasium environment.
    """
    env = gym.make(env_id)
    env = Monitor(env)
    return env


def make_vec_env(env_id: str = config.ENV_ID, n_envs: int = config.N_ENVS) -> VecEnv:
    """Create a vectorised (possibly parallel) environment.

    DummyVecEnv runs all envs sequentially in one process — ideal for
    a CPU laptop where spawning processes is slow.

    Args:
        env_id: Gymnasium environment ID string.
        n_envs: Number of parallel environments.

    Returns:
        A DummyVecEnv wrapping n_envs instances.
    """
    return DummyVecEnv([lambda: make_env(env_id) for _ in range(n_envs)])


def ensure_dirs() -> None:
    """Create output directories if they don't already exist."""
    for directory in [config.CHECKPOINT_DIR, config.ASSETS_DIR, "logs"]:
        os.makedirs(directory, exist_ok=True)


def plot_rewards(
    reward_history: list[float],
    save_path: str = config.REWARD_PLOT_PATH,
    window: int = 10,
) -> None:
    """Plot raw episode rewards and a smoothed rolling average.

    Args:
        reward_history: List of per-episode total rewards.
        save_path:      File path to save the PNG figure.
        window:         Rolling-average window size.
    """
    episodes = np.arange(1, len(reward_history) + 1)
    rewards = np.array(reward_history)

    # Rolling mean (pad with NaN so the first `window` points stay visible)
    smoothed = np.full_like(rewards, np.nan)
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed[window - 1:] = np.convolve(rewards, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.35, color="steelblue", label="Episode reward")
    ax.plot(episodes, smoothed, color="darkorange", linewidth=2,
            label=f"Rolling mean (window={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Training Reward Curve — Highway-v0 (PPO)", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Reward plot saved → {save_path}")
    