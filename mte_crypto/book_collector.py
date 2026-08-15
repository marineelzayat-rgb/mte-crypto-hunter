from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, deque
import gzip
import json
from pathlib import Path
import time

import websockets

from .flow import order_book_features


def _open_output(output: Path):
    if output.suffix == ".gz":
        return gzip.open(output, "at", encoding="utf-8")
    return output.open("a", encoding="utf-8")


async def collect(
    symbols: list[str],
    output: Path,
    *,
    duration_seconds: float | None = None,
    sample_interval_seconds: float = 1.0,
) -> None:
    """Record top-20 books plus executed taker flow; never places orders."""
    streams = "/".join(
        stream
        for symbol in symbols
        for stream in (f"{symbol.lower()}@depth20@100ms", f"{symbol.lower()}@aggTrade")
    )
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    tape: dict[str, deque[tuple[int, float, float]]] = defaultdict(deque)
    last_written_ns: dict[str, int] = defaultdict(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_flush = started
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as socket:
        with _open_output(output) as handle:
            async for raw in socket:
                if duration_seconds is not None and time.monotonic() - started >= duration_seconds:
                    break
                message = json.loads(raw)
                data = message.get("data", message)
                stream = message.get("stream", "")
                symbol = stream.split("@", 1)[0].upper()
                now_ns = time.time_ns()
                if stream.endswith("@aggTrade"):
                    quote = float(data["p"]) * float(data["q"])
                    signed_quote = -quote if bool(data["m"]) else quote
                    tape[symbol].append((now_ns, signed_quote, quote))
                    continue

                if now_ns - last_written_ns[symbol] < sample_interval_seconds * 1_000_000_000:
                    continue
                last_written_ns[symbol] = now_ns

                cutoff = now_ns - 60_000_000_000
                while tape[symbol] and tape[symbol][0][0] < cutoff:
                    tape[symbol].popleft()
                features = order_book_features(data["bids"], data["asks"])
                for seconds in (5, 30, 60):
                    start = now_ns - seconds * 1_000_000_000
                    recent = [item for item in tape[symbol] if item[0] >= start]
                    gross = sum(item[2] for item in recent)
                    features[f"tape_delta_ratio_{seconds}s"] = (
                        sum(item[1] for item in recent) / gross if gross else None
                    )
                    features[f"tape_quote_volume_{seconds}s"] = gross
                record = {
                    "received_time_ns": now_ns,
                    "symbol": symbol,
                    "last_update_id": data.get("lastUpdateId"),
                    **features,
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                if time.monotonic() - last_flush >= 15:
                    handle.flush()
                    last_flush = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Binance top-20 books for MTE forward tests")
    parser.add_argument("symbols", nargs="+", help="e.g. ACEUSDT SOLUSDT")
    parser.add_argument("--output", default="output/live_order_book.jsonl")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    args = parser.parse_args()
    asyncio.run(
        collect(
            [s.upper() for s in args.symbols],
            Path(args.output),
            duration_seconds=args.seconds,
            sample_interval_seconds=args.sample_seconds,
        )
    )


if __name__ == "__main__":
    main()
