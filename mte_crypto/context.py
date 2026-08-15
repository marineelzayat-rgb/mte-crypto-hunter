from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .research import load_csv


def attach_cross_section_context(
    trades: pd.DataFrame,
    data_dir: Path,
    min_quote_volume_24h: float = 1_000_000.0,
) -> pd.DataFrame:
    """Attach causal cross-sectional ranks to trade timestamps."""
    returns: dict[str, pd.Series] = {}
    volume_ratios: dict[str, pd.Series] = {}
    rvols: dict[str, pd.Series] = {}
    for path in sorted(data_dir.glob("*USDT.csv")):
        if path.stem.upper() == "BTCUSDT" or path.stat().st_size == 0:
            continue
        try:
            frame = load_csv(path)
        except (EmptyDataError, ValueError):
            continue
        if len(frame) < 300:
            continue
        qv24 = frame["quote_volume"].rolling(24, min_periods=18).sum()
        liquid = qv24 >= min_quote_volume_24h
        baseline = qv24.shift(24).rolling(24 * 20, min_periods=24 * 5).median()
        symbol = path.stem.upper()
        returns[symbol] = frame["close"].pct_change(24).where(liquid)
        volume_ratios[symbol] = (qv24 / baseline).where(liquid)
        rvols[symbol] = (
            frame["quote_volume"]
            / frame["quote_volume"].shift(1).rolling(168, min_periods=48).median()
        ).where(liquid)

    r24 = pd.concat(returns, axis=1)
    vr = pd.concat(volume_ratios, axis=1).reindex(r24.index)
    rv = pd.concat(rvols, axis=1).reindex(r24.index)
    return_rank = r24.rank(axis=1, pct=True)
    volume_rank = vr.rank(axis=1, pct=True)
    rvol_rank = rv.rank(axis=1, pct=True)
    median_return = r24.median(axis=1)

    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    keys = list(zip(out["entry_time"], out["symbol"]))
    out["entry_return_rank"] = [return_rank.at[t, s] if t in return_rank.index and s in return_rank.columns else float("nan") for t, s in keys]
    out["entry_volume_ratio_rank"] = [volume_rank.at[t, s] if t in volume_rank.index and s in volume_rank.columns else float("nan") for t, s in keys]
    out["entry_rvol_rank"] = [rvol_rank.at[t, s] if t in rvol_rank.index and s in rvol_rank.columns else float("nan") for t, s in keys]
    out["entry_return_vs_alt_median"] = [
        float(out.iloc[i]["entry_return_24h"] - median_return.get(t, float("nan")))
        for i, (t, _) in enumerate(keys)
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach causal cross-sectional context to trades")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trades = pd.read_csv(args.trades)
    out = attach_cross_section_context(trades, Path(args.data_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out[["symbol", "entry_time", "r_multiple", "entry_return_rank", "entry_volume_ratio_rank", "entry_rvol_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
