from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

import pandas as pd


KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


class BinancePublicClient:
    """Small dependency-free client for Binance public Spot data."""

    def __init__(self, base_url: str, timeout: float = 20.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _get(self, path: str, params: dict | None = None):
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "MTE-Crypto-Hunter/0.1"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_error = exc
                time.sleep(0.6 * (2 ** attempt))
        raise RuntimeError(f"Binance request failed after {self.retries} attempts: {url}") from last_error

    def exchange_info(self) -> dict:
        return self._get("/api/v3/exchangeInfo")

    def tickers_24h(self) -> list[dict]:
        return self._get("/api/v3/ticker/24hr")

    def klines(self, symbol: str, interval: str = "1h", limit: int = 1000, closed_only: bool = True) -> pd.DataFrame:
        raw = self._get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
        )
        frame = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        if frame.empty:
            return frame
        numeric = [c for c in KLINE_COLUMNS if c not in {"open_time", "close_time", "ignore"}]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        frame = frame.set_index("open_time").sort_index()
        if closed_only:
            frame = frame[frame["close_time"] <= pd.Timestamp.now(tz="UTC")]
        return frame


def liquid_usdt_universe(
    client: BinancePublicClient,
    min_quote_volume: float,
    top: int,
    explicit_symbols: Iterable[str] | None = None,
) -> list[dict]:
    info = client.exchange_info()
    status = {
        item["symbol"]: item
        for item in info.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed", True)
    }
    tickers = {item["symbol"]: item for item in client.tickers_24h()}
    excluded_suffixes = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
    excluded_bases = {"BTC", "USDC", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "AEUR"}

    requested = {s.upper() for s in explicit_symbols or []}
    rows: list[dict] = []
    for symbol, meta in status.items():
        if requested and symbol not in requested:
            continue
        if symbol.endswith(excluded_suffixes) or meta.get("baseAsset") in excluded_bases:
            continue
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        quote_volume = float(ticker.get("quoteVolume") or 0.0)
        if not requested and quote_volume < min_quote_volume:
            continue
        rows.append(
            {
                "symbol": symbol,
                "base_asset": meta.get("baseAsset"),
                "quote_volume_24h": quote_volume,
                "return_24h": float(ticker.get("priceChangePercent") or 0.0) / 100.0,
                "last_price": float(ticker.get("lastPrice") or 0.0),
            }
        )
    rows.sort(key=lambda row: row["quote_volume_24h"], reverse=True)
    return rows[:top] if not requested else rows
