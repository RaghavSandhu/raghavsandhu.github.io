"""Technical indicator computation and feature engineering.

All indicators are computed with pandas/numpy — no external TA library needed.
"""

import numpy as np
import pandas as pd


# ── Helper functions ──────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(close: pd.Series, window: int = 20, std_dev: float = 2.0):
    middle = _sma(close, window)
    std = close.rolling(window).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower)
    return upper, middle, lower, width, pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = ((high - prev_high).clip(lower=0)).where(
        (high - prev_high) > (prev_low - low), 0
    )
    minus_dm = ((prev_low - low).clip(lower=0)).where(
        (prev_low - low) > (high - prev_high), 0
    )
    atr = _atr(high, low, close, window)
    plus_di = 100 * _ema(plus_dm, window) / atr.replace(0, np.nan)
    minus_di = 100 * _ema(minus_dm, window) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = _ema(dx, window)
    return adx_val, plus_di, minus_di


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                window: int = 14, smooth: int = 3):
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(smooth).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = _sma(tp, window)
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)


def _roc(close: pd.Series, window: int = 12) -> pd.Series:
    return (close - close.shift(window)) / close.shift(window).replace(0, np.nan) * 100


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    delta = tp.diff()
    pos_flow = rmf.where(delta > 0, 0).rolling(window).sum()
    neg_flow = rmf.where(delta <= 0, 0).rolling(window).sum()
    mfi = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))
    return mfi


# ── Main API ──────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 40+ technical indicators as features to OHLCV dataframe."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = _sma(close, w)
        df[f"EMA_{w}"] = _ema(close, w)

    macd_line, macd_signal, macd_hist = _macd(close)
    df["MACD"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    adx_val, plus_di, minus_di = _adx(high, low, close)
    df["ADX"] = adx_val
    df["ADX_pos"] = plus_di
    df["ADX_neg"] = minus_di

    df["CCI"] = _cci(high, low, close)

    # Momentum
    df["RSI"] = _rsi(close)
    stoch_k, stoch_d = _stochastic(high, low, close)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d
    df["Williams_R"] = _williams_r(high, low, close)
    df["ROC"] = _roc(close)

    # Volatility
    bb_upper, bb_mid, bb_lower, bb_width, bb_pct = _bollinger_bands(close)
    df["BB_upper"] = bb_upper
    df["BB_middle"] = bb_mid
    df["BB_lower"] = bb_lower
    df["BB_width"] = bb_width
    df["BB_pct"] = bb_pct
    df["ATR"] = _atr(high, low, close)

    # Volume
    df["OBV"] = _obv(close, volume)
    df["MFI"] = _mfi(high, low, close, volume)

    # Price-derived
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"]
    df["price_sma20_ratio"] = close / df["SMA_20"]
    df["price_sma50_ratio"] = close / df["SMA_50"]

    # Candle patterns
    df["body_size"] = abs(df["Close"] - df["Open"]) / df["Open"]
    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]

    return df


def create_targets(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """Create prediction target columns.

    Args:
        df: DataFrame with at least a 'Close' column.
        horizon: Number of candles ahead to predict.
        threshold: Minimum % move to classify as up/down.
    """
    df = df.copy()
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target_return"] = future_return
    df["target_direction"] = 0
    df.loc[future_return > threshold, "target_direction"] = 1
    df.loc[future_return < -threshold, "target_direction"] = -1
    return df


def prepare_dataset(
    df: pd.DataFrame,
    sequence_length: int = 20,
    horizon: int = 3,
    threshold: float = 0.001,
    test_ratio: float = 0.2,
):
    """Full pipeline: indicators -> targets -> train/test split.

    Returns:
        (feature_cols, X_train, X_test, y_train_dir, y_test_dir,
         y_train_ret, y_test_ret, train_dates, test_dates, scaler)
    """
    from sklearn.preprocessing import StandardScaler

    df = add_technical_indicators(df)
    df = create_targets(df, horizon=horizon, threshold=threshold)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    exclude = {"Open", "High", "Low", "Close", "Volume",
               "target_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y_dir = df["target_direction"].values
    y_ret = df["target_return"].values
    dates = df.index

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train_dir, y_test_dir = y_dir[:split], y_dir[split:]
    y_train_ret, y_test_ret = y_ret[:split], y_ret[split:]
    train_dates, test_dates = dates[:split], dates[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        feature_cols,
        X_train, X_test,
        y_train_dir, y_test_dir,
        y_train_ret, y_test_ret,
        train_dates, test_dates,
        scaler,
    )


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20):
    """Reshape flat features into sequences for LSTM/Transformer."""
    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        sequences.append(X[i - seq_len: i])
        targets.append(y[i])
    return np.array(sequences), np.array(targets)
