from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import DEFAULT_CONFIG
from .features import enrich_hourly
from .research import load_csv
from .rule_eval import rules


def simulate_symbol(
    symbol: str,
    frame: pd.DataFrame,
    signal: pd.Series,
    fee_slippage_bps_each_side: float = 15.0,
    max_hold_h: int = 24 * 30,
    max_entry_return_24h: float | None = 0.50,
    allow_overlapping_entries: bool = False,
) -> list[dict]:
    trades: list[dict] = []
    signal_values = signal.fillna(False).to_numpy(dtype=bool)
    last_pivot_low = frame["pivot_low_confirmed"].ffill()
    cost = fee_slippage_bps_each_side / 10_000.0
    i = 0
    while i < len(frame) - 1:
        if not signal_values[i]:
            i += 1
            continue
        if max_entry_return_24h is not None and frame["return_24h_calc"].iloc[i] > max_entry_return_24h:
            i += 1
            continue
        pivot = last_pivot_low.iloc[i]
        atr = frame["atr"].iloc[i]
        if not np.isfinite(pivot) or not np.isfinite(atr):
            i += 1
            continue
        entry_raw = float(frame["close"].iloc[i])
        entry = entry_raw * (1.0 + cost)
        stop = float(pivot - 0.30 * atr)
        risk_pct = (entry - stop) / entry
        if stop <= 0 or risk_pct < 0.01 or risk_pct > 0.30:
            i += 1
            continue

        exit_i = min(i + max_hold_h, len(frame) - 1)
        exit_reason = "TIME/END"
        exit_raw = float(frame["close"].iloc[exit_i])
        mfe = 0.0
        mae = 0.0
        active_stop = stop

        for j in range(i + 1, exit_i + 1):
            high = float(frame["high"].iloc[j])
            low = float(frame["low"].iloc[j])
            open_ = float(frame["open"].iloc[j])
            close = float(frame["close"].iloc[j])
            mfe = max(mfe, high / entry - 1.0)
            mae = min(mae, low / entry - 1.0)

            # Intrabar stop is evaluated before a stop can be raised at the
            # current close. Gap behavior is conservative.
            if low <= active_stop:
                exit_i = j
                exit_raw = min(open_, active_stop)
                exit_reason = "STRUCTURAL_STOP"
                break

            new_pivot = frame["pivot_low_confirmed"].iloc[j]
            if np.isfinite(new_pivot):
                candidate = float(new_pivot - 0.30 * frame["atr"].iloc[j])
                if candidate < close:
                    active_stop = max(active_stop, candidate)

            # Earlier close-based H1 structure failure, matching the MTE idea
            # that a confirmed opposite swing break invalidates the direction.
            pivot_now = last_pivot_low.iloc[j]
            if np.isfinite(pivot_now) and close < pivot_now - 0.10 * frame["atr"].iloc[j]:
                exit_i = j
                exit_raw = close
                exit_reason = "H1_FLIP"
                break

        exit_price = exit_raw * (1.0 - cost)
        return_pct = exit_price / entry - 1.0
        trades.append(
            {
                "symbol": symbol,
                "entry_time": frame.index[i],
                "exit_time": frame.index[exit_i],
                "entry": entry,
                "initial_stop": stop,
                "initial_risk_pct": risk_pct,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "return_pct": return_pct,
                "r_multiple": return_pct / risk_pct,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "hold_h": exit_i - i,
                "entry_return_24h": frame["return_24h_calc"].iloc[i],
                "entry_rvol": frame["rvol_1h"].iloc[i],
                "entry_volume_24h_ratio": frame["volume_24h_ratio"].iloc[i],
                "entry_quote_volume_24h": frame["quote_volume_24h_roll"].iloc[i],
                "entry_relative_strength_24h": frame["relative_strength_24h"].iloc[i],
                "entry_btc_return_24h": frame["btc_return_24h"].iloc[i],
                "entry_btc_return_7d": frame["btc_return_7d"].iloc[i],
                "entry_green_volume_share_24h": frame["green_volume_share_24h"].iloc[i],
                "entry_taker_buy_ratio_24h": frame["taker_buy_ratio_24h"].iloc[i],
                "entry_range_position_7d": frame["range_position_7d"].iloc[i],
                "entry_last_spike_retention": frame["last_spike_retention"].iloc[i],
                "entry_prior_spike_count_7d": frame["prior_spike_count_7d"].iloc[i],
                "entry_compression_rank": frame["compression_rank"].iloc[i],
                "entry_break_distance_atr": frame["break_distance_atr"].iloc[i],
                "entry_tr_atr": frame["tr_atr"].iloc[i],
                "entry_body_ratio": frame["body_ratio"].iloc[i],
                "entry_bull_clv": frame["bull_clv"].iloc[i],
            }
        )
        i = i + 1 if allow_overlapping_entries else exit_i + 1
    return trades


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wins = trades["r_multiple"] > 0
    gross_profit = trades.loc[wins, "r_multiple"].sum()
    gross_loss = -trades.loc[~wins, "r_multiple"].sum()
    ordered = trades.sort_values("exit_time")
    equity_r = ordered["r_multiple"].cumsum()
    drawdown_r = equity_r.cummax() - equity_r
    top5 = trades.nlargest(min(5, len(trades)), "r_multiple")["r_multiple"].sum()
    total_r = trades["r_multiple"].sum()
    return {
        "trades": len(trades),
        "win_rate": wins.mean(),
        "profit_factor_r": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "total_r": total_r,
        "median_r": trades["r_multiple"].median(),
        "mean_r": trades["r_multiple"].mean(),
        "max_dd_r_naive": drawdown_r.max(),
        "median_hold_h": trades["hold_h"].median(),
        "top5_profit_share": top5 / total_r if total_r > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trade-level structural backtest for MTE Crypto rule families")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--btc", required=True)
    parser.add_argument("--output", default="output/trade_backtest.csv")
    parser.add_argument("--fee-slippage-bps", type=float, default=15.0)
    parser.add_argument("--no-chase-24h", type=float, default=0.50)
    parser.add_argument("--rules", nargs="*", help="Optional rule names to evaluate")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    btc_path = Path(args.btc)
    btc = load_csv(btc_path)
    all_trades: list[dict] = []

    for path in sorted(data_dir.glob("*.csv")):
        if path.resolve() == btc_path.resolve() or path.stat().st_size == 0:
            continue
        try:
            raw = load_csv(path)
        except (EmptyDataError, ValueError):
            continue
        if len(raw) < 300:
            continue
        frame = enrich_hourly(raw, btc, DEFAULT_CONFIG)
        for name, signal in rules(frame).items():
            if args.rules and name not in set(args.rules):
                continue
            trades = simulate_symbol(
                path.stem.upper(),
                frame,
                signal,
                fee_slippage_bps_each_side=args.fee_slippage_bps,
                max_entry_return_24h=args.no_chase_24h,
            )
            for trade in trades:
                trade["rule"] = name
            all_trades.extend(trades)

    trades_frame = pd.DataFrame(all_trades)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trades_frame.to_csv(output, index=False)
    summaries = []
    if not trades_frame.empty:
        for name, group in trades_frame.groupby("rule"):
            row = {"rule": name}
            row.update(summarize(group))
            summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("profit_factor_r", ascending=False)
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
