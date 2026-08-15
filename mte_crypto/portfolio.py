from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    starting_equity: float = 100_000.0
    risk_per_trade: float = 0.0025
    max_positions: int = 4


def simulate_portfolio(trades: pd.DataFrame, cfg: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a simple concurrency/risk cap to already-causal trade results.

    Realized PnL is booked at each exit.  Candidates sharing an entry timestamp
    are ordered by spot quote volume, a pre-entry execution-quality variable.
    The drawdown is therefore a closed-equity estimate, not intrabar MTM.
    """
    if trades.empty:
        return trades.copy(), pd.DataFrame()
    work = trades.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True)
    work["exit_time"] = pd.to_datetime(work["exit_time"], utc=True)
    work = work.sort_values(
        ["entry_time", "entry_quote_volume_24h"], ascending=[True, False]
    )

    equity = cfg.starting_equity
    peak = equity
    active: list[dict] = []
    accepted: list[dict] = []
    curve: list[dict] = []

    def realize_through(timestamp: pd.Timestamp) -> None:
        nonlocal equity, peak, active
        exiting = sorted((p for p in active if p["exit_time"] <= timestamp), key=lambda p: p["exit_time"])
        for position in exiting:
            equity += position["pnl"]
            peak = max(peak, equity)
            curve.append(
                {
                    "timestamp": position["exit_time"],
                    "equity": equity,
                    "drawdown_pct": equity / peak - 1.0,
                    "symbol": position["symbol"],
                }
            )
        active = [p for p in active if p["exit_time"] > timestamp]

    for entry_time, batch in work.groupby("entry_time", sort=True):
        realize_through(entry_time)
        slots = max(cfg.max_positions - len(active), 0)
        for _, trade in batch.head(slots).iterrows():
            risk_cash = equity * cfg.risk_per_trade
            pnl = risk_cash * float(trade["r_multiple"])
            position = {
                "symbol": trade["symbol"],
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "risk_cash": risk_cash,
                "pnl": pnl,
            }
            active.append(position)
            accepted.append({**trade.to_dict(), "risk_cash": risk_cash, "portfolio_pnl": pnl})

    if active:
        realize_through(max(p["exit_time"] for p in active))
    return pd.DataFrame(accepted), pd.DataFrame(curve).sort_values("timestamp")


def summarize(accepted: pd.DataFrame, curve: pd.DataFrame, cfg: PortfolioConfig) -> dict:
    if accepted.empty:
        return {"accepted_trades": 0}
    ending = cfg.starting_equity + accepted["portfolio_pnl"].sum()
    return {
        "accepted_trades": len(accepted),
        "ending_equity": ending,
        "return_pct": ending / cfg.starting_equity - 1.0,
        "max_closed_equity_dd_pct": -float(curve["drawdown_pct"].min()) if not curve.empty else 0.0,
        "risk_per_trade_pct": cfg.risk_per_trade,
        "max_positions": cfg.max_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply portfolio risk limits to an MTE trade file")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.0025)
    parser.add_argument("--max-positions", type=int, default=4)
    args = parser.parse_args()
    cfg = PortfolioConfig(args.starting_equity, args.risk_per_trade, args.max_positions)
    trades = pd.read_csv(args.trades)
    accepted, curve = simulate_portfolio(trades, cfg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output, index=False)
    curve.to_csv(output.with_name(output.stem + "_equity.csv"), index=False)
    summary = pd.DataFrame([summarize(accepted, curve, cfg)])
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
