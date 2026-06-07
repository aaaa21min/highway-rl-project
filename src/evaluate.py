"""
evaluate.py
-----------
Loads each saved checkpoint (untrained, half-trained, fully trained)
and records a short video of the agent playing in highway-v0.

Run AFTER training is complete:
    python src/evaluate.py

Output videos are saved to:
    videos/1_untrained-episode-0.mp4
    videos/2_half_trained-episode-0.mp4
    videos/3_fully_trained-episode-0.mp4
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import highway_env  # noqa: F401 — registers highway envs with gymnasium
from gymnasium.wrappers import RecordVideo

import config
from model import load_ppo_model
from utils import ensure_dirs


VIDEO_DIR: str = "videos"
EPISODES_PER_STAGE: int = 3   # how many episodes to record per checkpoint


def record_agent(
    checkpoint_path: str,
    video_prefix: str,
    episodes: int = EPISODES_PER_STAGE,
) -> None:
    """Load a checkpoint and record the agent playing for N episodes.

    Args:
        checkpoint_path: Path to the .zip model file (without extension).
        video_prefix:    Filename prefix for the saved video.
        episodes:        Number of episodes to record.
    """
    print(f"\n{'─' * 55}")
    print(f"  Recording: {video_prefix}")
    print(f"  Checkpoint: {checkpoint_path}.zip")
    print(f"{'─' * 55}")

    # Create env with rgb_array rendering so frames can be captured
    env = gym.make(config.ENV_ID, render_mode="rgb_array")

    # RecordVideo saves an MP4 automatically (one per episode)
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        name_prefix=video_prefix,
        episode_trigger=lambda ep: True,
    )

    model = load_ppo_model(checkpoint_path, env=env)

    for ep in range(episodes):
        obs, _ = env.reset()
        done: bool = False
        total_reward: float = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        print(f"  Episode {ep + 1}/{episodes} — reward: {total_reward:.2f}")

    env.close()
    print(f"  ✓ Video saved to {VIDEO_DIR}/{video_prefix}-*")


def evaluate() -> None:
    """Record videos for all three training stages."""

    ensure_dirs()
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("=" * 55)
    print("  CMP4501 — Evaluation & Video Recording")
    print("=" * 55)

    stages: list[tuple[str, str]] = [
        (config.CHECKPOINT_UNTRAINED, "1_untrained"),
        (config.CHECKPOINT_HALF,      "2_half_trained"),
        (config.CHECKPOINT_FULL,      "3_fully_trained"),
    ]

    for checkpoint_path, video_prefix in stages:
        zip_path = f"{checkpoint_path}.zip"
        if not os.path.exists(zip_path):
            print(f"\n[evaluate] WARNING: {zip_path} not found — skipping.")
            continue
        record_agent(checkpoint_path, video_prefix)

    print("\n" + "=" * 55)
    print("  All videos recorded! ✓")
    print(f"  Check the '{VIDEO_DIR}/' folder for your MP4 files.")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()