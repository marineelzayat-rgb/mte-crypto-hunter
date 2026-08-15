from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import DEFAULT_CONFIG
from .features import enrich_hourly


FEATURE_COLUMNS = [
    "rvol_1h",
    "volume_24h_ratio",
    "compression_rank",
    "relative_strength_24h",
    "distance_to_macro_high_atr",
    "break_distance_atr",
    "tr_atr",
    "body_ratio",
    "bull_clv",
    "return_24h_calc",
    "prior_spike_count_7d",
    "last_spike_retention",
    "range_position_7d",
    "drawdown_from_7d_high",
    "taker_buy_ratio_24h",
    "green_volume_share_24h",
    "structural_risk_pct",
]


def forward_max_return(close: pd.Series, high: pd.Series, hours: int) -> pd.Series:
    # Shift before reversing so the current bar is excluded from its own outcome.
    future = high.shift(-1)[::-1].rolling(hours, min_periods=hours).max()[::-1]
    return future / close - 1.0


def label_explosions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["forward_max_24h"] = forward_max_return(out["close"], out["high"], 24)
    out["forward_max_48h"] = forward_max_return(out["close"], out["high"], 48)
    # Outcome timestamp: the first bar where the move has actually become an
    # objective explosion. Features are sampled at fixed offsets BEFORE it.
    out["trailing_return_24h"] = out["high"] / out["close"].shift(24) - 1.0
    out["trailing_return_48h"] = out["high"] / out["close"].shift(48) - 1.0
    out["explosion"] = (out["trailing_return_24h"] >= 0.40) | (out["trailing_return_48h"] >= 0.60)
    previous_event = out["explosion"].shift(1).rolling(72, min_periods=1).max().fillna(False).astype(bool)
    out["event_start"] = out["explosion"] & ~previous_event
    return out


def event_study(symbol_frames: dict[str, pd.DataFrame], btc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels: list[pd.DataFrame] = []
    for symbol, frame in symbol_frames.items():
        enriched = enrich_hourly(frame, btc, DEFAULT_CONFIG)
        labeled = label_explosions(enriched)
        labeled["symbol"] = symbol
        panels.append(labeled)
    panel = pd.concat(panels).sort_index() if panels else pd.DataFrame()
    if panel.empty:
        return panel, pd.DataFrame()

    leads = [4, 12, 24, 48, 72]
    snapshots: list[dict] = []
    rng = np.random.default_rng(17)
    for symbol, symbol_panel in panel.groupby("symbol", sort=False):
        symbol_panel = symbol_panel.sort_index()
        event_positions = np.flatnonzero(symbol_panel["event_start"].to_numpy())
        unsafe = symbol_panel["explosion"].rolling(145, center=True, min_periods=1).max().fillna(False).astype(bool)
        safe_positions = np.flatnonzero(~unsafe.to_numpy())
        for event_pos in event_positions:
            event_time = symbol_panel.index[event_pos]
            for lead in leads:
                position = event_pos - lead
                if position < 0:
                    continue
                source = symbol_panel.iloc[position]
                row = {"symbol": symbol, "event_time": event_time, "observation_time": symbol_panel.index[position], "lead_h": lead, "sample": "event"}
                row.update({feature: source.get(feature, np.nan) for feature in FEATURE_COLUMNS})
                snapshots.append(row)

                # Match a control from the same symbol and roughly the same
                # calendar regime, but outside any explosion neighborhood.
                candidates = safe_positions[
                    np.abs(safe_positions - position) <= 24 * 21
                ]
                if candidates.size:
                    control_pos = int(rng.choice(candidates))
                    control = symbol_panel.iloc[control_pos]
                    control_row = {"symbol": symbol, "event_time": event_time, "observation_time": symbol_panel.index[control_pos], "lead_h": lead, "sample": "control"}
                    control_row.update({feature: control.get(feature, np.nan) for feature in FEATURE_COLUMNS})
                    snapshots.append(control_row)

    observations = pd.DataFrame(snapshots)
    if observations.empty:
        return observations, pd.DataFrame()

    summary_rows: list[dict] = []
    for lead, lead_rows in observations.groupby("lead_h"):
        for feature in FEATURE_COLUMNS:
            event_values = lead_rows.loc[lead_rows["sample"] == "event", feature].dropna()
            control_values = lead_rows.loc[lead_rows["sample"] == "control", feature].dropna()
            summary_rows.append(
                {
                    "lead_h": lead,
                    "feature": feature,
                    "events_n": len(event_values),
                    "controls_n": len(control_values),
                    "event_median": event_values.median(),
                    "control_median": control_values.median(),
                    "median_difference": event_values.median() - control_values.median(),
                    "event_p75": event_values.quantile(0.75),
                    "control_p75": control_values.quantile(0.75),
                }
            )
    return observations, pd.DataFrame(summary_rows)


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    timestamp_col = "open_time" if "open_time" in frame.columns else "timestamp"
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    frame = frame.set_index(timestamp_col).sort_index()
    required = {"open", "high", "low", "close", "volume", "quote_volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "trades" not in frame:
        frame["trades"] = np.nan
    if "taker_buy_quote" not in frame:
        frame["taker_buy_quote"] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MTE Crypto pre-explosion event study")
    parser.add_argument("--data-dir", required=True, help="Directory of SYMBOL.csv hourly files")
    parser.add_argument("--btc", required=True, help="BTCUSDT hourly CSV")
    parser.add_argument("--output", default="output/event_study.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    btc = load_csv(Path(args.btc))
    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for path in data_dir.glob("*.csv"):
        if path.resolve() == Path(args.btc).resolve():
            continue
        try:
            frame = load_csv(path)
            if len(frame) >= 200:
                frames[path.stem.upper()] = frame
            else:
                skipped.append(path.name)
        except (EmptyDataError, ValueError):
            skipped.append(path.name)
    observations, summary = event_study(frames, btc)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    observations.to_csv(output.with_name(output.stem + "_observations.csv"))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {output.resolve()}")
    if skipped:
        print(f"Skipped {len(skipped)} empty/short/malformed files")


if __name__ == "__main__":
    main()
