from __future__ import annotations

import argparse
import io
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .binance import KLINE_COLUMNS, BinancePublicClient, liquid_usdt_universe
from .config import DEFAULT_CONFIG


S3_API = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_BASE = "https://data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
MONTH_RE = re.compile(r"-(\d{4}-\d{2})\.zip$")


def _xml_listing(prefix: str, delimiter: str | None = None) -> list[str]:
    marker: str | None = None
    output: list[str] = []
    while True:
        params = {"prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        url = S3_API + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "MTE-Crypto-Hunter/0.1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            root = ET.fromstring(response.read())
        if delimiter:
            output.extend(node.text or "" for node in root.findall("s3:CommonPrefixes/s3:Prefix", NS))
        else:
            output.extend(node.text or "" for node in root.findall("s3:Contents/s3:Key", NS))
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=NS) or "false").lower() == "true"
        if not truncated:
            break
        marker = root.findtext("s3:NextMarker", default=None, namespaces=NS)
        if not marker:
            break
    return output


def historical_usdt_symbols() -> list[str]:
    prefixes = _xml_listing("data/spot/monthly/klines/", delimiter="/")
    symbols = [prefix.rstrip("/").split("/")[-1] for prefix in prefixes]
    excluded = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
    return sorted(symbol for symbol in symbols if symbol.endswith("USDT") and not symbol.endswith(excluded))


def monthly_keys(symbol: str, start_month: str, end_month: str) -> list[str]:
    prefix = f"data/spot/monthly/klines/{symbol}/1h/"
    keys = _xml_listing(prefix)
    selected = []
    for key in keys:
        match = MONTH_RE.search(key)
        if match and start_month <= match.group(1) <= end_month and not key.endswith("CHECKSUM"):
            selected.append(key)
    return sorted(selected)


def _timestamp_to_datetime(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    # Binance Spot archives use microseconds for newer files and milliseconds
    # for older files. A multi-year file can contain BOTH units, so inference
    # must be per value rather than once for the whole column.
    result = pd.Series(pd.NaT, index=numeric.index, dtype="datetime64[ns, UTC]")
    micro = numeric >= 10**14
    milli = numeric.notna() & ~micro
    result.loc[micro] = pd.to_datetime(numeric.loc[micro], unit="us", utc=True, errors="coerce")
    result.loc[milli] = pd.to_datetime(numeric.loc[milli], unit="ms", utc=True, errors="coerce")
    return result


def download_symbol(symbol: str, start_month: str, end_month: str, output_dir: Path, skip_existing: bool = True) -> tuple[str, int]:
    destination = output_dir / f"{symbol}.csv"
    if skip_existing and destination.exists() and destination.stat().st_size > 0:
        with destination.open("rb") as handle:
            bars = max(sum(1 for _ in handle) - 1, 0)
        return symbol, bars
    frames: list[pd.DataFrame] = []
    for key in monthly_keys(symbol, start_month, end_month):
        request = urllib.request.Request(
            f"{ARCHIVE_BASE}/{key}",
            headers={"User-Agent": "MTE-Crypto-Hunter/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            with archive.open(csv_name) as handle:
                frame = pd.read_csv(handle, header=None, names=KLINE_COLUMNS)
        frames.append(frame)
    if not frames:
        return symbol, 0

    combined = pd.concat(frames, ignore_index=True)
    combined["open_time"] = _timestamp_to_datetime(combined["open_time"])
    combined["close_time"] = _timestamp_to_datetime(combined["close_time"])
    numeric = [c for c in KLINE_COLUMNS if c not in {"open_time", "close_time", "ignore"}]
    combined[numeric] = combined[numeric].apply(pd.to_numeric, errors="coerce")
    combined = combined.dropna(subset=["open_time", "open", "high", "low", "close"])
    combined = combined.drop_duplicates("open_time").sort_values("open_time")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return symbol, len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download causal Binance monthly 1h archives")
    parser.add_argument("--start", required=True, help="First month YYYY-MM")
    parser.add_argument("--end", required=True, help="Last month YYYY-MM")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--all-historical", action="store_true")
    parser.add_argument("--top-current", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", default="data/archive")
    parser.add_argument("--refresh", action="store_true", help="Redownload files that already exist")
    args = parser.parse_args()

    if args.symbols:
        symbols = sorted({symbol.upper() for symbol in args.symbols})
    elif args.all_historical:
        symbols = historical_usdt_symbols()
    elif args.top_current:
        client = BinancePublicClient(DEFAULT_CONFIG.api_base)
        rows = liquid_usdt_universe(client, 0.0, args.top_current)
        symbols = [row["symbol"] for row in rows]
    else:
        raise SystemExit("Choose --symbols, --all-historical, or --top-current")

    output_dir = Path(args.output_dir)
    failures: list[tuple[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_symbol, symbol, args.start, args.end, output_dir, not args.refresh): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, bars = future.result()
                completed += 1
                if completed % 25 == 0:
                    print(f"Processed {completed}/{len(symbols)}")
                if bars == 0:
                    failures.append((symbol, "no archive files in range"))
            except Exception as exc:
                failures.append((symbol, str(exc)))

    print(f"Processed {completed} symbols; files are in {output_dir.resolve()}")
    print(f"No-data/errors: {len(failures)}")
    for symbol, error in failures[:30]:
        print(f"  {symbol}: {error}")


if __name__ == "__main__":
    main()
