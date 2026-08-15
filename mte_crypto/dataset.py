from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import DEFAULT_CONFIG
from .features import enrich_hourly
from .research import label_explosions, load_csv
from .rule_eval import clustered_starts, rules


COIN_FEATURES = [
    "return_24h_calc",
    "quote_volume_24h_roll",
    "rvol_1h",
    "volume_24h_ratio",
    "relative_strength_24h",
    "btc_return_24h",
    "btc_return_7d",
    "compression_rank",
    "structural_risk_pct",
    "break_distance_atr",
    "distance_to_macro_high_atr",
    "tr_atr",
    "body_ratio",
    "bull_clv",
    "prior_spike_count_7d",
    "last_spike_retention",
    "range_position_7d",
    "taker_buy_ratio_24h",
    "green_volume_share_24h",
]

REGIME_FEATURES = [
    "breadth_positive_24h",
    "breadth_positive_7d",
    "breadth_breakout_7d",
    "breadth_volume_expansion",
    "median_alt_return_24h",
    "median_alt_return_7d",
    "p90_alt_return_24h",
    "p95_alt_return_24h",
    "share_alt_return_24h_gt_10pct",
    "tail_heat_72h",
]


def build_dataset(
    data_dir: Path,
    btc_path: Path,
    regime_path: Path | None,
    rule_name: str = "ARMED_IGNITION",
    cooldown_h: int = 24,
) -> pd.DataFrame:
    btc = load_csv(btc_path)
    regime = None
    if regime_path is not None:
        regime = pd.read_csv(regime_path, parse_dates=["timestamp"]).set_index("timestamp")
    rows: list[pd.DataFrame] = []
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
        mask = clustered_starts(rules(frame)[rule_name].fillna(False), cooldown_h)
        positions = np.flatnonzero(mask.to_numpy())
        if not positions.size:
            continue
        event_positions = np.flatnonzero(frame["event_start"].to_numpy())
        labels = [
            bool(event_positions[(event_positions > pos) & (event_positions <= pos + 48)].size)
            for pos in positions
        ]
        selected = frame.iloc[positions][COIN_FEATURES].copy()
        selected["symbol"] = path.stem.upper()
        selected["signal_time"] = selected.index
        selected["target_explosion_48h"] = np.asarray(labels, dtype=int)
        if regime is not None:
            selected = selected.join(regime[REGIME_FEATURES], how="left")
        rows.append(selected.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a causal MTE candidate-learning dataset")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--btc", required=True)
    parser.add_argument("--regime")
    parser.add_argument("--rule", default="ARMED_IGNITION")
    parser.add_argument("--cooldown-h", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = build_dataset(
        Path(args.data_dir),
        Path(args.btc),
        Path(args.regime) if args.regime else None,
        args.rule,
        args.cooldown_h,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    rate = data["target_explosion_48h"].mean() if not data.empty else float("nan")
    print(f"rows={len(data):,} positives={rate:.3%} saved={output.resolve()}")


if __name__ == "__main__":
    main()
