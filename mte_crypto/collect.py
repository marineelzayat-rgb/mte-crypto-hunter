from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .binance import BinancePublicClient, liquid_usdt_universe
from .config import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect recent hourly Binance Spot history")
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--min-volume", type=float, default=5_000_000.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", default="data/recent")
    args = parser.parse_args()

    cfg = replace(
        DEFAULT_CONFIG,
        max_universe=args.top,
        min_quote_volume_24h=args.min_volume,
        workers=args.workers,
    )
    client = BinancePublicClient(cfg.api_base)
    universe = liquid_usdt_universe(client, cfg.min_quote_volume_24h, cfg.max_universe)
    if not any(row["symbol"] == "BTCUSDT" for row in universe):
        universe.append({"symbol": "BTCUSDT"})

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []

    def fetch(symbol: str):
        frame = client.klines(symbol, cfg.interval, cfg.kline_limit)
        frame.reset_index().to_csv(output / f"{symbol}.csv", index=False)
        return symbol, len(frame)

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(fetch, row["symbol"]): row["symbol"] for row in universe}
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, bars = future.result()
                completed += 1
                if completed % 20 == 0:
                    print(f"Collected {completed}/{len(universe)} symbols")
                if bars < 200:
                    failures.append((symbol, f"only {bars} bars"))
            except Exception as exc:
                failures.append((symbol, str(exc)))

    print(f"Collected {completed} symbols into {output.resolve()}")
    if failures:
        print("Failures:")
        for symbol, error in failures:
            print(f"  {symbol}: {error}")


if __name__ == "__main__":
    main()

