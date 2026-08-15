from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .mte_ignition import compute_mte_ignitions
from .research import load_csv


def _load_history(symbol: str, directories: list[str]) -> pd.DataFrame:
    frames = []
    for directory in directories:
        path = Path(directory) / f"{symbol}.csv"
        if path.exists() and path.stat().st_size:
            frames.append(load_csv(path))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _first_passage(
    frame: pd.DataFrame,
    entry_i: int,
    raw_stop: float,
    fee_slippage_bps_each_side: float,
    max_hold_h: int,
) -> dict:
    cost = fee_slippage_bps_each_side / 10_000.0
    entry_raw = float(frame["close"].iloc[entry_i])
    entry = entry_raw * (1.0 + cost)
    risk_cash_per_unit = entry - raw_stop
    if risk_cash_per_unit <= 0:
        return {"valid": False}
    target3 = entry + 3.0 * risk_cash_per_unit
    target5 = entry + 5.0 * risk_cash_per_unit
    hit3_time = pd.NaT
    hit5_time = pd.NaT
    stop_time = pd.NaT
    fixed_exit_raw = float(frame["close"].iloc[min(entry_i + max_hold_h, len(frame) - 1)])
    fixed_reason = "TIME/END"
    final_i = min(entry_i + max_hold_h, len(frame) - 1)
    stopped = False
    for j in range(entry_i + 1, final_i + 1):
        open_ = float(frame["open"].iloc[j])
        low = float(frame["low"].iloc[j])
        high = float(frame["high"].iloc[j])
        # Conservative when stop and target occur inside the same hourly bar.
        if low <= raw_stop:
            stop_time = frame.index[j]
            stopped = True
            if pd.isna(hit3_time):
                fixed_exit_raw = min(open_, raw_stop)
                fixed_reason = "STOP_BEFORE_3R"
                final_i = j
            break
        if pd.isna(hit3_time) and high >= target3:
            hit3_time = frame.index[j]
            fixed_exit_raw = target3
            fixed_reason = "TARGET_3R"
            final_i = j
        if pd.isna(hit5_time) and high >= target5:
            hit5_time = frame.index[j]
            break
    fixed_exit = fixed_exit_raw * (1.0 - cost)
    fixed_r = (fixed_exit - entry) / risk_cash_per_unit
    return {
        "valid": True,
        "entry": entry,
        "stop": raw_stop,
        "risk_pct": risk_cash_per_unit / entry,
        "target_3r": target3,
        "target_5r": target5,
        "hit_3r": pd.notna(hit3_time),
        "hit_5r": pd.notna(hit5_time),
        "hit_3r_time": hit3_time,
        "hit_5r_time": hit5_time,
        "stop_time": stop_time,
        "fixed_3r_exit_reason": fixed_reason,
        "fixed_3r_r_multiple": fixed_r,
        "fixed_3r_hold_h": final_i - entry_i,
        "stopped_eventually": stopped,
    }


def _process_symbol(
    symbol: str,
    hunts: list[dict],
    directories: list[str],
    fee_slippage_bps_each_side: float,
    min_risk_pct: float,
    max_risk_pct: float,
    max_chase_return_24h: float,
    stop_mode: str,
) -> list[dict]:
    frame = _load_history(symbol, directories)
    if len(frame) < 500:
        return [{**hunt, "activation_status": "NO_DATA"} for hunt in hunts]
    mte = compute_mte_ignitions(frame)
    return24 = frame["close"].pct_change(24)
    outputs: list[dict] = []
    for hunt in hunts:
        hunt_time = pd.Timestamp(hunt["signal_time"])
        if hunt_time.tzinfo is None:
            hunt_time = hunt_time.tz_localize("UTC")
        expiry = hunt_time + pd.Timedelta(hours=48)
        positions = np.flatnonzero((frame.index >= hunt_time) & (frame.index <= expiry))
        if not positions.size:
            outputs.append({**hunt, "activation_status": "NO_WINDOW_DATA"})
            continue
        first_i = int(positions[0])
        cancel_low = float(mte["last_h1_pivot_low"].iloc[first_i])
        activated_i: int | None = None
        status = "EXPIRED"
        for i in positions:
            atr = float(mte["atr"].iloc[i])
            if np.isfinite(cancel_low) and float(frame["close"].iloc[i]) < cancel_low - 0.10 * atr:
                status = "CANCELLED_H1"
                break
            if not bool(mte["bull_ignition"].iloc[i]):
                continue
            stop_price = float(mte["structural_stop"].iloc[i])
            if stop_mode == "h1_pivot":
                pivot_low = float(mte["last_h1_pivot_low"].iloc[i])
                stop_price = pivot_low - 0.30 * atr if np.isfinite(pivot_low) else np.nan
            entry_with_cost = float(frame["close"].iloc[i]) * (1.0 + fee_slippage_bps_each_side / 10_000.0)
            risk_pct = (entry_with_cost - stop_price) / entry_with_cost if np.isfinite(stop_price) else np.nan
            if not (min_risk_pct <= risk_pct <= max_risk_pct):
                status = "IGNITION_RISK_REJECT"
                continue
            if float(return24.iloc[i]) > max_chase_return_24h:
                status = "IGNITION_NO_CHASE"
                continue
            activated_i = int(i)
            status = "ACTIVATED"
            break
        row = {
            **hunt,
            "activation_status": status,
            "hunt_time": hunt_time,
            "expiry_time": expiry,
            "cancel_low": cancel_low,
        }
        if activated_i is not None:
            row.update(
                {
                    "entry_time": frame.index[activated_i],
                    "entry_close": float(frame["close"].iloc[activated_i]),
                    "activation_delay_h": activated_i - first_i,
                    "broken_h4": int(mte["broken_h4"].iloc[activated_i]),
                    "broken_d1": int(mte["broken_d1"].iloc[activated_i]),
                    "trend_lock": int(mte["trend_lock"].iloc[activated_i]),
                    "activation_return_24h": float(return24.iloc[activated_i]),
                    "stop_mode": stop_mode,
                }
            )
            stop_price = float(mte["structural_stop"].iloc[activated_i])
            if stop_mode == "h1_pivot":
                stop_price = float(mte["last_h1_pivot_low"].iloc[activated_i]) - 0.30 * float(mte["atr"].iloc[activated_i])
            passage = _first_passage(
                frame,
                activated_i,
                stop_price,
                fee_slippage_bps_each_side,
                24 * 30,
            )
            row.update(passage)
        outputs.append(row)
    return outputs


def summarize(rows: pd.DataFrame) -> dict:
    activated = rows[rows["activation_status"] == "ACTIVATED"].copy()
    if activated.empty:
        return {"hunts": len(rows), "activated": 0}
    r = activated["fixed_3r_r_multiple"].dropna()
    wins = r > 0
    gross_profit = r[wins].sum()
    gross_loss = -r[~wins].sum()
    return {
        "hunts": len(rows),
        "activated": len(activated),
        "activation_rate": len(activated) / len(rows),
        "hit_3r": int(activated["hit_3r"].sum()),
        "hit_3r_rate": float(activated["hit_3r"].mean()),
        "hit_5r": int(activated["hit_5r"].sum()),
        "hit_5r_rate": float(activated["hit_5r"].mean()),
        "fixed_3r_pf": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "fixed_3r_total_r": float(r.sum()),
        "median_activation_delay_h": float(activated["activation_delay_h"].median()),
        "median_risk_pct": float(activated["risk_pct"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Hunter -> exact selected-line MTE ignition windows")
    parser.add_argument("--hunts", required=True)
    parser.add_argument("--history-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fee-slippage-bps", type=float, default=15.0)
    parser.add_argument("--min-risk-pct", type=float, default=0.01)
    parser.add_argument("--max-risk-pct", type=float, default=0.10)
    parser.add_argument("--max-chase-return-24h", type=float, default=0.35)
    parser.add_argument("--stop-mode", choices=["line", "h1_pivot"], default="line")
    args = parser.parse_args()

    hunts = pd.read_csv(args.hunts, parse_dates=["signal_time"])
    groups = {symbol: group.to_dict(orient="records") for symbol, group in hunts.groupby("symbol")}
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_symbol,
                symbol,
                records,
                args.history_dirs,
                args.fee_slippage_bps,
                args.min_risk_pct,
                args.max_risk_pct,
                args.max_chase_return_24h,
                args.stop_mode,
            ): symbol
            for symbol, records in groups.items()
        }
        completed = 0
        for future in as_completed(futures):
            rows.extend(future.result())
            completed += 1
            if completed % 20 == 0:
                print(f"Processed {completed}/{len(futures)} symbols", flush=True)
    result = pd.DataFrame(rows).sort_values("signal_time")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = pd.DataFrame([summarize(result)])
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
