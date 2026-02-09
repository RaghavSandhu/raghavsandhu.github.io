"""Entry point for the Stock Trend Predictor app.

Usage:
    streamlit run src/main.py
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.dashboard.app import main

if __name__ == "__main__":
    main()
