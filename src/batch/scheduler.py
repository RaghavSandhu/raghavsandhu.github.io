"""Market-hours scheduler.

Runs full training at market open (and hourly retrains),
then quick predictions every 3 minutes during market hours.

Usage:
    python -m src.batch.scheduler
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root on path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.batch.config import (
    WATCHLIST, TIMEZONE, REFRESH_INTERVAL_SECONDS,
    FULL_RETRAIN_INTERVAL_SECONDS,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
)
from src.batch.store import init_db, cleanup_old_data
from src.batch.runner import run_full_train, run_quick_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Shutdown signal received, stopping...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _get_et_now():
    """Get current time in US/Eastern."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TIMEZONE))
    except ImportError:
        # Fallback: assume UTC-5 (EST) if zoneinfo not available
        from datetime import timezone, timedelta
        return datetime.now(timezone(timedelta(hours=-5)))


def _is_market_hours(now=None) -> bool:
    """Check if current time is within market hours (M-F, 9:30-16:00 ET)."""
    if now is None:
        now = _get_et_now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
    return market_open <= now <= market_close


def _is_premarket(now=None) -> bool:
    """Check if within 15 min before market open (for pre-training)."""
    if now is None:
        now = _get_et_now()
    if now.weekday() >= 5:
        return False
    pre_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE - 15, second=0)
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    return pre_open <= now < market_open


def run_loop():
    """Main scheduler loop."""
    log.info("=" * 60)
    log.info("Stock Prediction Batch Scheduler started")
    log.info(f"Watchlist: {len(WATCHLIST)} stocks")
    log.info(f"Refresh: every {REFRESH_INTERVAL_SECONDS}s")
    log.info(f"Full retrain: every {FULL_RETRAIN_INTERVAL_SECONDS}s")
    log.info(f"Market hours: {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} - "
             f"{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE:02d} ET")
    log.info("=" * 60)

    init_db()
    cleanup_old_data(days=7)

    last_full_train = 0
    initial_train_done = False

    while _running:
        now = _get_et_now()

        if not _is_market_hours(now) and not _is_premarket(now):
            next_check = 60  # Check every minute outside market hours
            log.info(f"Outside market hours ({now.strftime('%H:%M ET, %A')}). "
                     f"Next check in {next_check}s.")
            time.sleep(next_check)
            continue

        current_time = time.time()

        # Full train: at start, pre-market, or every FULL_RETRAIN_INTERVAL
        need_full_train = (
            not initial_train_done
            or _is_premarket(now)
            or (current_time - last_full_train) >= FULL_RETRAIN_INTERVAL_SECONDS
        )

        if need_full_train:
            log.info("Running FULL TRAIN...")
            result = run_full_train(WATCHLIST)
            last_full_train = time.time()
            initial_train_done = True
            log.info(f"Full train: {result['ok']} OK, {result['error']} failed")
        else:
            log.info("Running QUICK PREDICT...")
            result = run_quick_predict(WATCHLIST)
            log.info(f"Quick predict: {result['ok']} OK, {result['error']} failed")

        # Wait for next cycle
        elapsed = time.time() - current_time
        sleep_time = max(0, REFRESH_INTERVAL_SECONDS - elapsed)
        if sleep_time > 0 and _running:
            log.info(f"Sleeping {sleep_time:.0f}s until next cycle...")
            # Sleep in small intervals to catch shutdown signals
            for _ in range(int(sleep_time)):
                if not _running:
                    break
                time.sleep(1)

    log.info("Scheduler stopped.")


if __name__ == "__main__":
    run_loop()
