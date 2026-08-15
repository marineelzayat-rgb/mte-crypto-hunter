from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .ranker import select_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a frozen MTE Hunter candidate dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    data = pd.read_csv(args.dataset, parse_dates=["signal_time"])
    selected = select_candidates(model, data, args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(
        f"selected={len(selected)} true={int(selected['target_explosion_48h'].sum())} "
        f"precision={selected['target_explosion_48h'].mean():.3f} saved={output.resolve()}"
    )


if __name__ == "__main__":
    main()
