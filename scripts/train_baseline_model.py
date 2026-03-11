#!/usr/bin/env python3
"""
Legacy entry point — delegates to the new research pipeline.

For full control, use scripts/train_model.py directly:
    python scripts/train_model.py --help

This script is kept for backward compatibility and runs the new pipeline
with default arguments, falling back to synthetic data if the DB is empty.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    train_model_script = SCRIPT_DIR / "train_model.py"
    cmd = [sys.executable, str(train_model_script)]

    # If no DB data will be available, add --synthetic flag
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        from app.research.dataset import load_features_df
        df = load_features_df(settings.database_url)
        if len(df) < 50:
            cmd.extend(["--synthetic", "--n-samples", "2000"])
    except Exception:
        cmd.extend(["--synthetic", "--n-samples", "2000"])

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
