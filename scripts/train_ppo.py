"""Command-line entry point for the CleanRL-style PPO trainer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic2d.ppo_cleanrl import main


if __name__ == "__main__":
    main()
