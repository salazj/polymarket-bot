#!/usr/bin/env python3
"""
Run the bot in DRY RUN mode (no real orders placed).

This is the default and safest way to test the system.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["DRY_RUN"] = "true"

from app.main import main

if __name__ == "__main__":
    main()
