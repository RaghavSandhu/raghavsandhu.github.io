"""Batch pipeline configuration — watchlist, timing, paths."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "results.db"

# ── Watchlist (100 liquid US stocks) ──────────────────────────────────
WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ORCL", "CRM",
    # Semiconductors
    "AMD", "INTC", "QCOM", "MU", "MRVL", "AMAT", "LRCX", "KLAC",
    "ON", "NXPI",
    # Software / Cloud
    "ADBE", "NOW", "PANW", "SNOW", "PLTR", "NET", "DDOG", "CRWD",
    "ZS", "MNDY",
    # Consumer / E-commerce
    "NFLX", "DIS", "CMCSA", "ABNB", "BKNG", "UBER", "LYFT", "DASH",
    "SPOT", "ROKU",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK", "V", "MA",
    # Payments / Fintech
    "PYPL", "SQ", "SOFI", "COIN", "HOOD", "AFRM", "NU", "MELI",
    "GDOT", "FIS",
    # Healthcare / Biotech
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "AMGN", "GILD",
    "MRNA", "BIIB",
    # Industrials / Energy
    "XOM", "CVX", "COP", "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX",
    # EV / Clean Energy
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FSLR", "ENPH", "PLUG",
    "CHPT", "QS",
    # Retail / Consumer
    "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
    "LULU", "GPS",
]

# ── Scheduling ────────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
TIMEZONE = "US/Eastern"

REFRESH_INTERVAL_SECONDS = 180  # 3 minutes
FULL_RETRAIN_INTERVAL_SECONDS = 3600  # Retrain models every 1 hour

# ── Model defaults ────────────────────────────────────────────────────
EPOCHS = 5
RL_EPISODES = 10
XGB_N_ESTIMATORS = 50
SEQ_LEN = 20
HORIZON = 3
THRESHOLD = 0.001
BATCH_SIZE = 64
MAX_WORKERS = 8  # Parallel stock processing threads

# ── Data fetch ────────────────────────────────────────────────────────
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "7d"
