"""Stock data fetching using yfinance."""

import yfinance as yf
import pandas as pd


VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "1d"]

# yfinance max period per interval
_MAX_PERIOD = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1d": "max",
}


def fetch_stock_data(
    ticker: str,
    period: str | None = None,
    interval: str = "5m",
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        period: Lookback period (e.g. "60d"). Defaults to max allowed for interval.
        interval: Candle interval. One of VALID_INTERVALS.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by datetime.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}")

    if period is None:
        period = _MAX_PERIOD[interval]

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} with period={period}, interval={interval}")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def get_stock_info(ticker: str) -> dict:
    """Return basic stock info (name, sector, etc.)."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "currency": info.get("currency", "USD"),
    }
