from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import DEFAULT_CONFIG
from .features import enrich_hourly
from .research import label_explosions, load_csv


def clustered_starts(mask: pd.Series, cooldown_h: int = 72) -> pd.Series:
    recent = mask.shift(1).rolling(cooldown_h, min_periods=1).max().fillna(False).astype(bool)
    return mask & ~recent


def rules(frame: pd.DataFrame) -> dict[str, pd.Series]:
    liquid = frame["quote_volume_24h_roll"] >= 5_000_000.0
    rs_positive = frame["relative_strength_24h"] >= 0.0
    close_to_break = frame["break_distance_atr"] >= -3.0
    quality = (frame["tr_atr"] >= 1.05) & (frame["body_ratio"] >= 0.55) & (frame["bull_clv"] >= 0.68)
    actual_break = frame["break_distance_atr"] >= DEFAULT_CONFIG.break_buffer_atr

    wakeup_near = liquid & frame["recent_wakeup"] & (frame["rvol_1h"] >= 1.5) & (frame["volume_24h_ratio"] >= 1.5) & close_to_break & rs_positive
    pre_ignition_hunt = (
        liquid
        & frame["recent_wakeup"]
        & frame["rvol_1h"].between(1.5, 10.0)
        & frame["volume_24h_ratio"].between(1.5, 10.0)
        & frame["break_distance_atr"].between(-3.0, DEFAULT_CONFIG.break_buffer_atr)
        & frame["return_24h_calc"].between(0.0, 0.20)
        & rs_positive
    )
    wakeup_quality = liquid & frame["recent_wakeup"] & (frame["rvol_1h"] >= 2.0) & (frame["volume_24h_ratio"] >= 1.5) & close_to_break & quality & rs_positive
    reignition = liquid & frame["recent_wakeup"] & (frame["rvol_1h"] >= 1.5) & (frame["volume_24h_ratio"] >= 1.2) & (frame["break_distance_atr"] >= -2.0) & rs_positive
    multi_spike_hold = (
        liquid
        & (frame["prior_spike_count_7d"] >= 2)
        & (frame["last_spike_retention"] >= 0.0)
        & (frame["range_position_7d"] >= 0.60)
        & (frame["rvol_1h"] >= 1.5)
        & (frame["volume_24h_ratio"] >= 1.2)
        & rs_positive
    )
    spot_pressure = (frame["taker_buy_ratio_24h"] >= 0.52) & (frame["green_volume_share_24h"] >= 0.55)
    early_liquid = (
        (frame["quote_volume_24h_roll"] >= 5_000_000.0)
        & actual_break & quality & rs_positive
        & frame["return_24h_calc"].between(0.05, 0.10)
        & frame["structural_risk_pct"].between(0.05, 0.10)
        & frame["rvol_1h"].between(2.0, 10.0)
        & frame["volume_24h_ratio"].between(1.5, 10.0)
    )
    # Frozen candidate selected for an untouched-year holdout.  Each extra
    # condition has a market interpretation: meaningful spot liquidity,
    # stored energy, broad green-candle participation, and no BTC crash.
    mte_crypto_v0_1 = (
        early_liquid
        & (frame["quote_volume_24h_roll"] >= 20_000_000.0)
        & (frame["compression_rank"] <= 0.20)
        & (frame["green_volume_share_24h"] >= 0.60)
        & (frame["btc_return_24h"] >= -0.02)
    )
    early_low_liquidity = (
        frame["quote_volume_24h_roll"].between(1_000_000.0, 5_000_000.0)
        & actual_break & quality
        & (frame["relative_strength_24h"] >= 0.08)
        & frame["return_24h_calc"].between(0.08, 0.25)
        & frame["structural_risk_pct"].between(0.10, 0.25)
        & frame["rvol_1h"].between(3.0, 15.0)
        & frame["volume_24h_ratio"].between(1.0, 2.5)
        & (frame["range_position_7d"] >= 0.85)
        & (frame["green_volume_share_24h"] >= 0.60)
    )
    return {
        "MTE_STRICT": liquid & actual_break & quality & (frame["rvol_1h"] >= 2.0),
        "WAKEUP_NEAR_BREAK": wakeup_near,
        "PRE_IGNITION_HUNT": pre_ignition_hunt,
        "WAKEUP_QUALITY": wakeup_quality,
        "ARMED_IGNITION": wakeup_quality & actual_break,
        "REIGNITION": reignition,
        "REIGNITION_BREAK": reignition & actual_break & quality,
        "MULTI_SPIKE_HOLD": multi_spike_hold,
        "MULTI_SPIKE_IGNITION": multi_spike_hold & actual_break & quality,
        "SPOT_PRESSURE_IGNITION": multi_spike_hold & actual_break & quality & spot_pressure,
        "EARLY_LIQUID_IGNITION": early_liquid,
        "MTE_CRYPTO_V0_1": mte_crypto_v0_1,
        "EARLY_LOW_LIQUIDITY": early_low_liquidity,
        "VOLUME_ACCEL": liquid & (frame["rvol_1h"] >= 2.0) & (frame["volume_24h_ratio"] >= 1.5) & close_to_break & rs_positive,
    }


def evaluate_directory(data_dir: Path, btc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    btc = load_csv(btc_path)
    per_rule = {name: {"events": 0, "captured": 0, "signals": 0, "true_signals": 0, "lead_h": []} for name in rules(enrich_hourly(btc, btc, DEFAULT_CONFIG))}
    signal_rows: list[dict] = []

    for path in sorted(data_dir.glob("*.csv")):
        if path.resolve() == btc_path.resolve() or path.stat().st_size == 0:
            continue
        try:
            raw = load_csv(path)
        except (EmptyDataError, ValueError):
            continue
        if len(raw) < 300:
            continue
        frame = label_explosions(enrich_hourly(raw, btc, DEFAULT_CONFIG))
        event_positions = np.flatnonzero(frame["event_start"].to_numpy())

        for name, raw_signal in rules(frame).items():
            signal = clustered_starts(raw_signal.fillna(False), 72)
            signal_positions = np.flatnonzero(signal.to_numpy())
            stats = per_rule[name]
            stats["events"] += len(event_positions)
            stats["signals"] += len(signal_positions)

            for event_pos in event_positions:
                earlier = signal_positions[(signal_positions >= event_pos - 48) & (signal_positions < event_pos)]
                if earlier.size:
                    stats["captured"] += 1
                    stats["lead_h"].append(int(event_pos - earlier[-1]))

            for signal_pos in signal_positions:
                later = event_positions[(event_positions > signal_pos) & (event_positions <= signal_pos + 48)]
                is_true = bool(later.size)
                stats["true_signals"] += int(is_true)
                row = frame.iloc[signal_pos]
                signal_rows.append(
                    {
                        "rule": name,
                        "symbol": path.stem.upper(),
                        "signal_time": frame.index[signal_pos],
                        "true_within_48h": is_true,
                        "lead_h": int(later[0] - signal_pos) if is_true else np.nan,
                        "price": row["close"],
                        "rvol_1h": row["rvol_1h"],
                        "volume_24h_ratio": row["volume_24h_ratio"],
                        "relative_strength_24h": row["relative_strength_24h"],
                        "break_distance_atr": row["break_distance_atr"],
                    }
                )

    summary = []
    for name, stats in per_rule.items():
        events = stats["events"]
        signals = stats["signals"]
        summary.append(
            {
                "rule": name,
                "events": events,
                "captured_events": stats["captured"],
                "event_recall": stats["captured"] / events if events else np.nan,
                "signals": signals,
                "true_signals": stats["true_signals"],
                "signal_precision": stats["true_signals"] / signals if signals else np.nan,
                "median_lead_h": np.median(stats["lead_h"]) if stats["lead_h"] else np.nan,
            }
        )
    return pd.DataFrame(summary).sort_values("signal_precision", ascending=False), pd.DataFrame(signal_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate causal MTE Crypto rule families")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--btc", required=True)
    parser.add_argument("--output", default="output/rule_evaluation.csv")
    args = parser.parse_args()

    summary, signals = evaluate_directory(Path(args.data_dir), Path(args.btc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    signals.to_csv(output.with_name(output.stem + "_signals.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
