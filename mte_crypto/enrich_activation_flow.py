from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .flow import activation_flow_features
from .hunter_mte_backtest import _load_history


def enrich_results(rows: pd.DataFrame, history_dirs: list[str]) -> pd.DataFrame:
    out = rows.copy()
    flow_rows: list[dict] = []
    for symbol, group in out[out["activation_status"] == "ACTIVATED"].groupby("symbol"):
        frame = _load_history(symbol, history_dirs)
        if frame.empty:
            continue
        for index, row in group.iterrows():
            timestamp = pd.Timestamp(row["entry_time"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            positions = np.flatnonzero(frame.index == timestamp)
            if positions.size:
                flow_rows.append({"_index": index, **activation_flow_features(frame, int(positions[0]))})
    if not flow_rows:
        return out
    flow = pd.DataFrame(flow_rows).set_index("_index")
    return out.join(flow)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach causal activation-time Binance flow features")
    parser.add_argument("--input", required=True)
    parser.add_argument("--history-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.input)
    enriched = enrich_results(rows, args.history_dirs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    print(f"wrote {len(enriched)} rows to {output}")


if __name__ == "__main__":
    main()
