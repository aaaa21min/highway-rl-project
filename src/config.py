"""
config.py
---------
Central configuration for the CMP4501 Highway-Env RL project.
All hyperparameters and paths are defined here, keeping them separate
from training logic (required by the project rubric).
"""

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR: str = "checkpoints"
ASSETS_DIR: str = "assets"

CHECKPOINT_UNTRAINED: str = f"{CHECKPOINT_DIR}/agent_untrained"
CHECKPOINT_HALF: str = f"{CHECKPOINT_DIR}/agent_half_trained"
CHECKPOINT_FULL: str = f"{CHECKPOINT_DIR}/agent_fully_trained"

REWARD_PLOT_PATH: str = f"{ASSETS_DIR}/reward_plot.png"

# ── Environment ────────────────────────────────────────────────────────────
ENV_ID: str = "highway-v0"

# Number of parallel environments (1 = single env, safe for CPU laptops)
N_ENVS: int = 1

# ── PPO Hyperparameters ────────────────────────────────────────────────────
# Total training steps (raise to ~200_000 for a fully trained agent;
# keep low here for Stage 1 smoke-test).
TOTAL_TIMESTEPS: int = 100_000

# Steps collected per update cycle across all envs
N_STEPS: int = 256

# Mini-batch size for gradient updates
BATCH_SIZE: int = 64

# Number of epochs over collected data per update
N_EPOCHS: int = 10

# Discount factor (how much to value future rewards)
GAMMA: float = 0.99

# Learning rate
LEARNING_RATE: float = 3e-4

# Clipping range for PPO surrogate objective
CLIP_RANGE: float = 0.2

# ── Logging ────────────────────────────────────────────────────────────────
TENSORBOARD_LOG_DIR: str = "logs"
