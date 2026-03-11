#!/usr/bin/env python3
"""
Run the bot in LIVE mode.

⚠️  WARNING: This will place REAL orders with REAL money.
⚠️  Ensure you have:
⚠️    1. Tested thoroughly in dry-run mode
⚠️    2. Set appropriate risk limits in .env
⚠️    3. Derived and configured API credentials
⚠️    4. Reviewed RISK_CONTROLS.md

Usage:
    python scripts/run_live_bot.py --markets will-x-happen
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["DRY_RUN"] = "false"
os.environ["ENABLE_LIVE_TRADING"] = "true"
os.environ["LIVE_TRADING_ACKNOWLEDGED"] = "true"


def main() -> None:
    from app.config import get_settings

    settings = get_settings()

    if not settings.has_credentials:
        print("ERROR: API credentials not configured. Run scripts/derive_api_credentials.py first.")
        sys.exit(1)

    print("=" * 60)
    print("WARNING: LIVE TRADING MODE")
    print("=" * 60)
    print(f"Strategy: {settings.strategy}")
    print(f"Max position/market: ${settings.max_position_per_market}")
    print(f"Max total exposure: ${settings.max_total_exposure}")
    print(f"Max daily loss: ${settings.max_daily_loss}")
    print(f"Default order size: ${settings.default_order_size}")
    print(f"DRY_RUN: {settings.dry_run}")
    print(f"ENABLE_LIVE_TRADING: {settings.enable_live_trading}")
    print(f"LIVE_TRADING_ACKNOWLEDGED: {settings.live_trading_acknowledged}")
    print("=" * 60)

    confirmation = input("Type 'I ACCEPT THE RISK' to proceed with live trading: ")
    if confirmation != "I ACCEPT THE RISK":
        print("Aborted.")
        sys.exit(0)

    from app.main import main as bot_main
    bot_main()


if __name__ == "__main__":
    main()
