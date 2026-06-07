"""
train.py
--------
Full training pipeline with checkpoints at three stages:
  1. Untrained    — random weights, saved before any training
  2. Half-trained — saved after 50% of timesteps
  3. Fully trained — saved after 100% of timesteps

Run from the project root:
    python src/train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import config
from model import build_ppo_model
from utils import ensure_dirs, make_vec_env, plot_rewards


class RewardLoggerCallback(BaseCallback):
    """Collects per-episode rewards during training.

    Stable-Baselines3 exposes episode info via `self.locals["infos"]`.
    Each entry may contain an `"episode"` key (added by the Monitor
    wrapper) with the cumulative episode reward.

    Args:
        verbose: 1 = print each episode reward, 0 = silent.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Called after every env step. Returns True to continue training."""
        infos: list[dict] = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                ep_reward: float = float(episode_info["r"])
                self.episode_rewards.append(ep_reward)
                if self.verbose:
                    ep = len(self.episode_rewards)
                    print(f"  Episode {ep:>4d} | Reward: {ep_reward:+.2f}")
        return True


def train() -> None:
    """Three-stage training pipeline with checkpoint saving."""

    ensure_dirs()
    half_steps: int = config.TOTAL_TIMESTEPS // 2

    print("=" * 60)
    print("  CMP4501 — Highway-Env PPO Training")
    print(f"  Total timesteps : {config.TOTAL_TIMESTEPS:,}")
    print(f"  Half-way point  : {half_steps:,}")
    print("=" * 60)

    # 1. Environment
    print("\n[train] Creating environment …")
    env = make_vec_env(config.ENV_ID, config.N_ENVS)

    # 2. Model
    print("[train] Building PPO model …")
    model = build_ppo_model(env)

    # ── STAGE 1: Save UNTRAINED checkpoint ────────────────────────────────
    print(f"\n[train] Saving UNTRAINED checkpoint → {config.CHECKPOINT_UNTRAINED}")
    model.save(config.CHECKPOINT_UNTRAINED)
    print("[train] ✓ Untrained checkpoint saved (random weights)")

    # ── STAGE 2: Train first half → save HALF-TRAINED checkpoint ──────────
    print(f"\n[train] Training first half ({half_steps:,} timesteps) …")
    callback = RewardLoggerCallback(verbose=1)
    model.learn(
        total_timesteps=half_steps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=True,
    )
    print(f"\n[train] Saving HALF-TRAINED checkpoint → {config.CHECKPOINT_HALF}")
    model.save(config.CHECKPOINT_HALF)
    print("[train] ✓ Half-trained checkpoint saved")

    # ── STAGE 3: Train second half → save FULLY TRAINED checkpoint ─────────
    print(f"\n[train] Training second half ({half_steps:,} timesteps) …")
    model.learn(
        total_timesteps=half_steps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    print(f"\n[train] Saving FULLY TRAINED checkpoint → {config.CHECKPOINT_FULL}")
    model.save(config.CHECKPOINT_FULL)
    print("[train] ✓ Fully trained checkpoint saved")

    # ── Reward plot ────────────────────────────────────────────────────────
    rewards: list[float] = callback.episode_rewards
    if rewards:
        print(f"\n[train] {len(rewards)} total episodes completed.")
        print(f"[train] Avg reward : {np.mean(rewards):.2f}")
        print(f"[train] Min reward : {np.min(rewards):.2f}")
        print(f"[train] Max reward : {np.max(rewards):.2f}")
        plot_rewards(rewards)
    else:
        print("[train] No complete episodes — increase TOTAL_TIMESTEPS.")

    env.close()

    print("\n[train] All done! ✓")
    print("  Checkpoints saved:")
    print(f"    {config.CHECKPOINT_UNTRAINED}.zip")
    print(f"    {config.CHECKPOINT_HALF}.zip")
    print(f"    {config.CHECKPOINT_FULL}.zip")
    print(f"  Reward plot : {config.REWARD_PLOT_PATH}")


if __name__ == "__main__":
    train()