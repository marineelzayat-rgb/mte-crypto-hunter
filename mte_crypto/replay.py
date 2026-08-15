from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .binance import BinancePublicClient
from .config import DEFAULT_CONFIG
from .features import enrich_hourly


def ignition_mask(enriched: pd.DataFrame) -> pd.Series:
    cfg = DEFAULT_CONFIG
    return (
        (enriched["close"] > enriched["breakout_level"] + cfg.break_buffer_atr * enriched["atr"])
        & (enriched["tr_atr"] >= cfg.min_expansion_tr_atr)
        & (enriched["body_ratio"] >= cfg.min_expansion_body_ratio)
        & (enriched["bull_clv"] >= cfg.min_expansion_clv)
        & (enriched["rvol_1h"] >= cfg.min_ignition_rvol)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay causal ignition timestamps for one symbol")
    parser.add_argument("symbol")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    client = BinancePublicClient(DEFAULT_CONFIG.api_base)
    frame = client.klines(symbol, DEFAULT_CONFIG.interval, DEFAULT_CONFIG.kline_limit)
    btc = client.klines("BTCUSDT", DEFAULT_CONFIG.interval, DEFAULT_CONFIG.kline_limit)
    enriched = enrich_hourly(frame, btc, DEFAULT_CONFIG)
    signals = enriched.loc[
        ignition_mask(enriched),
        [
            "open", "high", "low", "close", "rvol_1h", "volume_24h_ratio",
            "return_24h_calc", "breakout_level", "break_distance_atr",
            "tr_atr", "body_ratio", "bull_clv", "compression_rank",
            "relative_strength_24h",
        ],
    ].copy()
    print(signals.tail(30).to_string(float_format=lambda x: f"{x:.4f}"))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(output)
        print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()

