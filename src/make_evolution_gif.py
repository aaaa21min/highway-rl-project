
"""
make_evolution_gif.py
---------------------
Stitches the 3 stage videos (untrained, half-trained, fully trained)
into a single labelled evolution GIF for the README.

Run AFTER evaluate.py has produced the videos:
    python src/make_evolution_gif.py

Output:
    assets/evolution.gif
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import imageio.v2 as imageio
import numpy as np

from utils import ensure_dirs

VIDEO_DIR: str = "videos"
OUTPUT_PATH: str = "assets/evolution.gif"

# Stage label → filename glob pattern
STAGES: list[tuple[str, str]] = [
    ("UNTRAINED",     "1_untrained-episode-0.mp4"),
    ("HALF-TRAINED",  "2_half_trained-episode-0.mp4"),
    ("FULLY TRAINED", "3_fully_trained-episode-0.mp4"),
]

FPS: int = 15
MAX_FRAMES_PER_STAGE: int = 90   # ~6 seconds per stage at 15 fps


def load_video_frames(video_path: str, max_frames: int) -> list[np.ndarray]:
    """Read frames from an MP4 video file, capped at max_frames."""
    reader = imageio.get_reader(video_path)
    frames: list[np.ndarray] = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        frames.append(np.asarray(frame))
    reader.close()
    return frames


def find_first_match(pattern: str) -> str | None:
    """Find the first file matching the glob pattern (or None)."""
    matches = sorted(glob.glob(os.path.join(VIDEO_DIR, pattern)))
    return matches[0] if matches else None


def main() -> None:
    ensure_dirs()

    print("=" * 55)
    print("  Building evolution GIF")
    print("=" * 55)

    all_frames: list[np.ndarray] = []

    for label, pattern in STAGES:
        # If a literal filename doesn't exist, try a more permissive glob
        path = os.path.join(VIDEO_DIR, pattern)
        if not os.path.exists(path):
            prefix = pattern.split("-")[0]
            path = find_first_match(f"{prefix}-*.mp4")
            if path is None:
                print(f"[!] No video found for stage '{label}' — skipping.")
                continue

        print(f"[+] Loading {label:14s} ← {path}")
        frames = load_video_frames(path, MAX_FRAMES_PER_STAGE)
        all_frames.extend(frames)

    if not all_frames:
        print("[!] No frames collected. Run evaluate.py first.")
        return

    print(f"\n[+] Writing GIF ({len(all_frames)} frames) → {OUTPUT_PATH}")
    imageio.mimsave(OUTPUT_PATH, all_frames, fps=FPS, loop=0)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"[✓] Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
    