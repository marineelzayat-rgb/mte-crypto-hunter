from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import simulate_symbol
from .config import DEFAULT_CONFIG
from .features import enrich_hourly
from .research import load_csv


def attach_trade_outcomes(dataset: pd.DataFrame, data_dir: Path, btc_path: Path) -> pd.DataFrame:
    btc = load_csv(btc_path)
    outcomes: list[dict] = []
    work = dataset.copy()
    work["signal_time"] = pd.to_datetime(work["signal_time"], utc=True)
    for symbol, group in work.groupby("symbol"):
        path = data_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = enrich_hourly(load_csv(path), btc, DEFAULT_CONFIG)
        signal = pd.Series(frame.index.isin(group["signal_time"]), index=frame.index)
        trades = simulate_symbol(symbol, frame, signal, allow_overlapping_entries=True)
        outcomes.extend(trades)
    outcome_frame = pd.DataFrame(outcomes)
    if outcome_frame.empty:
        return work.iloc[0:0]
    outcome_frame["entry_time"] = pd.to_datetime(outcome_frame["entry_time"], utc=True)
    keep = [
        "symbol", "entry_time", "exit_time", "r_multiple", "return_pct",
        "mfe_pct", "mae_pct", "hold_h", "exit_reason",
    ]
    return work.merge(
        outcome_frame[keep],
        left_on=["symbol", "signal_time"],
        right_on=["symbol", "entry_time"],
        how="inner",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach independent structural trade outcomes to MTE candidates")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--btc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.dataset)
    out = attach_trade_outcomes(data, Path(args.data_dir), Path(args.btc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"rows={len(out):,} mean_R={out['r_multiple'].mean():.3f} saved={output.resolve()}")


if __name__ == "__main__":
    main()
