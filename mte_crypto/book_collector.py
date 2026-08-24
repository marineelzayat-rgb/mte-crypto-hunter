from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, deque
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import time

import websockets

from .flow import order_book_features


ORDER_BOOK_MIN_FREE_BYTES = 128 * 1024 * 1024
ORDER_BOOK_MAX_RAW_BYTES = 8 * 1024 * 1024


def _merge_metric(target: dict, value: float) -> None:
    target["count"] = int(target.get("count", 0)) + 1
    target["sum"] = float(target.get("sum", 0.0)) + value
    target["min"] = min(float(target.get("min", value)), value)
    target["max"] = max(float(target.get("max", value)), value)
    target["positive"] = int(target.get("positive", 0)) + int(value > 0)


def _merge_book_summary(target: dict, source: dict) -> dict:
    target.setdefault("version", 1)
    target_symbols = target.setdefault("symbols", {})
    for symbol, incoming in (source.get("symbols") or {}).items():
        current = target_symbols.setdefault(symbol, {"samples": 0, "metrics": {}})
        current["samples"] = int(current.get("samples", 0)) + int(incoming.get("samples", 0))
        current_metrics = current.setdefault("metrics", {})
        for name, values in (incoming.get("metrics") or {}).items():
            metric = current_metrics.setdefault(
                name,
                {"count": 0, "sum": 0.0, "min": values["min"], "max": values["max"], "positive": 0},
            )
            metric["count"] += int(values.get("count", 0))
            metric["sum"] += float(values.get("sum", 0.0))
            metric["min"] = min(float(metric["min"]), float(values["min"]))
            metric["max"] = max(float(metric["max"]), float(values["max"]))
            metric["positive"] += int(values.get("positive", 0))
    return target


def summarize_order_book_file(path: Path) -> dict:
    """Reduce raw snapshots to mergeable per-symbol distribution statistics."""
    summary: dict = {"version": 1, "source": path.name, "symbols": {}}
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                symbol = str(record.get("symbol") or "")
                if not symbol:
                    continue
                bucket = summary["symbols"].setdefault(symbol, {"samples": 0, "metrics": {}})
                bucket["samples"] += 1
                for name, raw in record.items():
                    if name in {"symbol", "received_time_ns", "last_update_id"}:
                        continue
                    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                        continue
                    value = float(raw)
                    if math.isfinite(value):
                        _merge_metric(bucket["metrics"].setdefault(name, {}), value)
                if line_number % 1000 == 0:
                    # Let the status-server thread answer Railway health requests
                    # while a large gzip is being compacted.
                    time.sleep(0.001)
    except (EOFError, gzip.BadGzipFile):
        # Disk exhaustion can truncate the final gzip member.  Keep every
        # complete snapshot decoded before the damaged tail.
        summary["truncated_source"] = True
    return summary


def reclaim_order_book_storage(
    data_dir: Path,
    *,
    min_free_bytes: int = ORDER_BOOK_MIN_FREE_BYTES,
    max_raw_bytes: int = ORDER_BOOK_MAX_RAW_BYTES,
) -> list[str]:
    """Compact raw books near capacity while retaining research distributions."""
    raw_dir = data_dir / "order_book"
    if not raw_dir.exists():
        return []
    files = sorted(raw_dir.glob("*.jsonl.gz"), key=lambda item: item.stat().st_mtime)
    total_raw_bytes = sum(item.stat().st_size for item in files)
    compacted: list[str] = []
    summary_dir = data_dir / "order_book_summary"
    runtime_dir = Path(os.getenv("MTE_RUNTIME_DIR", "/tmp/mte-runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        try:
            free = shutil.disk_usage(data_dir).free
            raw_size = path.stat().st_size
            if (
                free >= min_free_bytes
                and raw_size <= max_raw_bytes
                and total_raw_bytes <= max_raw_bytes
            ):
                continue
            incoming = summarize_order_book_file(path)
            summary_path = summary_dir / f"{path.name.removesuffix('.jsonl.gz')}.json"
            existing = {}
            try:
                if summary_path.exists():
                    existing = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
            merged = _merge_book_summary(existing, incoming)
            staged = runtime_dir / f"{summary_path.name}.tmp"
            staged.write_text(json.dumps(merged, separators=(",", ":")))
            path.unlink()
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(staged.read_text())
            staged.unlink()
            compacted.append(path.name)
            total_raw_bytes = max(0, total_raw_bytes - raw_size)
        except (OSError, EOFError, ValueError) as exc:
            print(
                f"Order-book compaction failed for {path.name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            break
    return compacted


def collection_timeout_seconds(duration_seconds: float | None) -> float | None:
    if duration_seconds is None:
        return None
    duration = max(0.0, float(duration_seconds))
    grace = min(10.0, max(0.25, duration * 0.10))
    return duration + grace


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
    symbol_sample_intervals: dict[str, float] | None = None,
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
    # ``async for`` only advances when Binance sends a message.  Previously the
    # duration check lived solely inside that loop, so a silent-but-open socket
    # could freeze the entire scanner indefinitely.  The outer timeout is an
    # absolute deadline and therefore also covers connection/close stalls.
    absolute_timeout = collection_timeout_seconds(duration_seconds)
    try:
        async with asyncio.timeout(absolute_timeout):
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=10,
                close_timeout=5,
            ) as socket:
                with _open_output(output) as handle:
                    async for raw in socket:
                        if (
                            duration_seconds is not None
                            and time.monotonic() - started >= duration_seconds
                        ):
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

                        interval = max(
                            0.25,
                            float(
                                (symbol_sample_intervals or {}).get(
                                    symbol, sample_interval_seconds
                                )
                            ),
                        )
                        if (
                            now_ns - last_written_ns[symbol]
                            < interval * 1_000_000_000
                        ):
                            continue
                        last_written_ns[symbol] = now_ns

                        cutoff = now_ns - 60_000_000_000
                        while tape[symbol] and tape[symbol][0][0] < cutoff:
                            tape[symbol].popleft()
                        features = order_book_features(data["bids"], data["asks"])
                        for seconds in (5, 30, 60):
                            start = now_ns - seconds * 1_000_000_000
                            recent = [
                                item for item in tape[symbol] if item[0] >= start
                            ]
                            gross = sum(item[2] for item in recent)
                            features[f"tape_delta_ratio_{seconds}s"] = (
                                sum(item[1] for item in recent) / gross
                                if gross
                                else None
                            )
                            features[f"tape_quote_volume_{seconds}s"] = gross
                        record = {
                            "received_time_ns": now_ns,
                            "symbol": symbol,
                            "sample_interval_seconds": interval,
                            "last_update_id": data.get("lastUpdateId"),
                            **features,
                        }
                        handle.write(
                            json.dumps(record, separators=(",", ":")) + "\n"
                        )
                        if time.monotonic() - last_flush >= 15:
                            handle.flush()
                            last_flush = time.monotonic()
    except TimeoutError:
        # Normal bounded completion when the stream did not yield in time.
        return


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
