from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .research import load_csv


def build_regime_from_frames(frames: dict[str, pd.DataFrame], min_quote_volume_24h: float = 1_000_000.0) -> pd.DataFrame:
    """Build the same causal breadth series from already-loaded live frames."""
    ret24: dict[str, pd.Series] = {}
    ret7d: dict[str, pd.Series] = {}
    break7d: dict[str, pd.Series] = {}
    volume_expansion: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        if symbol.upper() == "BTCUSDT" or len(frame) < 300:
            continue
        qv24 = frame["quote_volume"].rolling(24, min_periods=18).sum()
        liquid = qv24 >= min_quote_volume_24h
        baseline = qv24.shift(24).rolling(24 * 20, min_periods=24 * 5).median()
        prior_high = frame["high"].shift(1).rolling(168, min_periods=72).max()
        ret24[symbol] = frame["close"].pct_change(24).where(liquid)
        ret7d[symbol] = frame["close"].pct_change(168).where(liquid)
        break7d[symbol] = frame["close"].gt(prior_high).where(liquid)
        volume_expansion[symbol] = qv24.ge(1.5 * baseline).where(liquid)
        active[symbol] = liquid
    return _assemble_regime(ret24, ret7d, break7d, volume_expansion, active)


def _assemble_regime(
    ret24: dict[str, pd.Series],
    ret7d: dict[str, pd.Series],
    break7d: dict[str, pd.Series],
    volume_expansion: dict[str, pd.Series],
    active: dict[str, pd.Series],
) -> pd.DataFrame:
    if not ret24:
        return pd.DataFrame()
    r24 = pd.concat(ret24, axis=1)
    r7d = pd.concat(ret7d, axis=1).reindex(r24.index)
    brk = pd.concat(break7d, axis=1).reindex(r24.index)
    vol = pd.concat(volume_expansion, axis=1).reindex(r24.index)
    act = pd.concat(active, axis=1).reindex(r24.index).astype("boolean").fillna(False).astype(bool)
    count = act.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=r24.index)
    out["active_alt_count"] = count
    out["breadth_positive_24h"] = r24.gt(0).where(act).sum(axis=1) / count
    out["breadth_positive_7d"] = r7d.gt(0).where(act).sum(axis=1) / count
    out["breadth_breakout_7d"] = brk.where(act).sum(axis=1) / count
    out["breadth_volume_expansion"] = vol.where(act).sum(axis=1) / count
    out["median_alt_return_24h"] = r24.median(axis=1, skipna=True)
    out["median_alt_return_7d"] = r7d.median(axis=1, skipna=True)
    out["p90_alt_return_24h"] = r24.quantile(0.90, axis=1)
    out["p95_alt_return_24h"] = r24.quantile(0.95, axis=1)
    out["share_alt_return_24h_gt_10pct"] = r24.gt(0.10).where(act).sum(axis=1) / count
    out["tail_heat_72h"] = out["p95_alt_return_24h"].shift(1).rolling(72, min_periods=12).max()
    return out[out["active_alt_count"] >= 20]


def build_regime(data_dir: Path, min_quote_volume_24h: float = 1_000_000.0) -> pd.DataFrame:
    """Build causal cross-sectional alt-market breadth from historical members."""
    ret24: dict[str, pd.Series] = {}
    ret7d: dict[str, pd.Series] = {}
    break7d: dict[str, pd.Series] = {}
    volume_expansion: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}

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
        prior_high = frame["high"].shift(1).rolling(168, min_periods=72).max()
        symbol = path.stem.upper()
        ret24[symbol] = frame["close"].pct_change(24).where(liquid)
        ret7d[symbol] = frame["close"].pct_change(168).where(liquid)
        break7d[symbol] = frame["close"].gt(prior_high).where(liquid)
        volume_expansion[symbol] = qv24.ge(1.5 * baseline).where(liquid)
        active[symbol] = liquid

    return _assemble_regime(ret24, ret7d, break7d, volume_expansion, active)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a causal Binance alt-market regime series")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-volume", type=float, default=1_000_000.0)
    args = parser.parse_args()
    regime = build_regime(Path(args.data_dir), args.min_volume)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    regime.to_csv(output, index_label="timestamp")
    print(regime.tail().to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"Saved {len(regime):,} rows: {output.resolve()}")


if __name__ == "__main__":
    main()
