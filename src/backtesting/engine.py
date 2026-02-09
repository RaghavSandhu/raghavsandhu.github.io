"""Backtesting engine for evaluating trading strategies."""

import numpy as np
import pandas as pd


class BacktestEngine:
    """Simulate trading on historical data using model predictions or RL actions."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        position_size: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> dict:
        """Run backtest.

        Args:
            prices: Array of closing prices.
            signals: Array of signals — 1=buy, -1=sell, 0=hold.
            dates: Optional datetime index for the equity curve.

        Returns:
            Dict with equity curve, trade log, and performance metrics.
        """
        n = min(len(prices), len(signals))
        prices = prices[:n]
        signals = signals[:n]

        capital = self.initial_capital
        position = 0  # 1=long, -1=short, 0=flat
        entry_price = 0.0
        equity_curve = [capital]
        trades = []
        returns_list = []

        for i in range(1, n):
            signal = int(signals[i])
            price = prices[i]

            # Open or flip position
            if signal == 1 and position != 1:
                if position == -1:
                    # Close short
                    pnl = (entry_price - price) / entry_price - self.transaction_cost
                    pnl *= self.position_size
                    capital *= 1 + pnl
                    trades.append({
                        "type": "close_short", "entry": entry_price,
                        "exit": price, "pnl_pct": pnl,
                    })
                    returns_list.append(pnl)
                # Open long
                position = 1
                entry_price = price
                trades.append({"type": "open_long", "entry": price, "exit": None, "pnl_pct": None})

            elif signal == -1 and position != -1:
                if position == 1:
                    # Close long
                    pnl = (price - entry_price) / entry_price - self.transaction_cost
                    pnl *= self.position_size
                    capital *= 1 + pnl
                    trades.append({
                        "type": "close_long", "entry": entry_price,
                        "exit": price, "pnl_pct": pnl,
                    })
                    returns_list.append(pnl)
                # Open short
                position = -1
                entry_price = price
                trades.append({"type": "open_short", "entry": price, "exit": None, "pnl_pct": None})

            # Mark-to-market equity
            if position == 1:
                unrealized = (price - entry_price) / entry_price * self.position_size
            elif position == -1:
                unrealized = (entry_price - price) / entry_price * self.position_size
            else:
                unrealized = 0.0
            equity_curve.append(capital * (1 + unrealized))

        # Force close at end
        if position != 0:
            price = prices[-1]
            if position == 1:
                pnl = (price - entry_price) / entry_price - self.transaction_cost
            else:
                pnl = (entry_price - price) / entry_price - self.transaction_cost
            pnl *= self.position_size
            capital *= 1 + pnl
            returns_list.append(pnl)
            equity_curve[-1] = capital

        # Compute metrics
        equity = np.array(equity_curve)
        total_return = (equity[-1] / equity[0]) - 1
        daily_returns = np.diff(equity) / equity[:-1]

        metrics = self._compute_metrics(equity, daily_returns, returns_list, total_return)
        metrics["total_trades"] = len([t for t in trades if t["pnl_pct"] is not None])

        result = {
            "equity_curve": equity,
            "trades": trades,
            "metrics": metrics,
        }

        if dates is not None:
            result["dates"] = dates[:len(equity)]

        return result

    def _compute_metrics(self, equity, daily_returns, trade_returns, total_return):
        trade_returns = np.array(trade_returns) if trade_returns else np.array([0])
        winning = trade_returns[trade_returns > 0]
        losing = trade_returns[trade_returns <= 0]

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # Sharpe ratio (annualized, assume 252 trading days)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = float(np.sum(winning)) if len(winning) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losing))) if len(losing) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss

        return {
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "final_capital": float(equity[-1]),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown * 100),
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "win_rate": float(len(winning) / max(len(trade_returns), 1)),
            "avg_win": float(np.mean(winning)) if len(winning) > 0 else 0.0,
            "avg_loss": float(np.mean(losing)) if len(losing) > 0 else 0.0,
            "total_trades": len(trade_returns),
        }


def compare_strategies(backtest_results: dict[str, dict]) -> pd.DataFrame:
    """Compare multiple strategy backtest results side by side.

    Args:
        backtest_results: Dict of {"strategy_name": backtest_result_dict}.

    Returns:
        DataFrame with metrics as rows and strategies as columns.
    """
    records = {}
    for name, result in backtest_results.items():
        records[name] = result["metrics"]
    return pd.DataFrame(records).T
