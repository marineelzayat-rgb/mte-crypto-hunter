from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

from .book_collector import collect
from .config import DEFAULT_CONFIG
from .monitor import format_alert, send_telegram
from .scan import scan_market


UTC = timezone.utc
ALERT_STATE = "HUNTER_ALERT"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def update_active_candidates(
    rows: list[dict],
    active: dict[str, dict],
    previous_states: dict[str, str],
    *,
    now: datetime,
    ttl_hours: int,
) -> tuple[dict[str, dict], dict[str, str], list[dict]]:
    active = {
        symbol: item
        for symbol, item in active.items()
        if datetime.fromisoformat(item["expires_at"]) > now
    }
    new_alerts: list[dict] = []
    next_states: dict[str, str] = {}
    for row in rows:
        symbol = str(row.get("symbol", ""))
        state = str(row.get("state", ""))
        if not symbol:
            continue
        next_states[symbol] = state
        if state == ALERT_STATE and previous_states.get(symbol) != ALERT_STATE:
            active[symbol] = {
                "detected_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=ttl_hours)).isoformat(),
                "hunter_probability": row.get("hunter_probability"),
                "price": row.get("price"),
            }
            new_alerts.append(row)
    return active, next_states, new_alerts


def run_scan(data_dir: Path, top: int, ttl_hours: int) -> list[str]:
    now = _utc_now()
    frame = scan_market(replace(DEFAULT_CONFIG, max_universe=top))
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "latest_scan.csv", index=False)
    frame.to_csv(data_dir / f"scan_{now:%Y%m%d_%H%M%S}.csv", index=False)

    active_path = data_dir / "active_candidates.json"
    states_path = data_dir / "scanner_states.json"
    active, states, new_alerts = update_active_candidates(
        frame.to_dict(orient="records"),
        _read_json(active_path),
        _read_json(states_path),
        now=now,
        ttl_hours=ttl_hours,
    )
    _write_json(active_path, active)
    _write_json(states_path, states)
    for row in new_alerts:
        message = format_alert(row)
        if not send_telegram(message):
            print(message, flush=True)
    return sorted(active)


def main() -> None:
    data_dir = Path(os.getenv("MTE_DATA_DIR", "output/live"))
    top = int(os.getenv("MTE_SCAN_TOP", "120"))
    interval_seconds = max(300, int(os.getenv("MTE_SCAN_INTERVAL_SECONDS", "3600")))
    ttl_hours = max(1, int(os.getenv("MTE_CANDIDATE_TTL_HOURS", "48")))
    max_book_symbols = max(1, int(os.getenv("MTE_MAX_BOOK_SYMBOLS", "10")))
    sample_seconds = max(0.25, float(os.getenv("MTE_BOOK_SAMPLE_SECONDS", "1")))

    print(
        f"MTE daemon started: top={top}, scan_every={interval_seconds}s, "
        f"ttl={ttl_hours}h, data={data_dir}",
        flush=True,
    )
    if os.environ.get("MTE_TELEGRAM_BOT_TOKEN") and os.environ.get(
        "MTE_TELEGRAM_CHAT_ID"
    ):
        try:
            if send_telegram(
                "MTE Crypto Hunter is online. Scanning the top "
                f"{top} Binance Spot USDT altcoins every "
                f"{interval_seconds // 60} minutes."
            ):
                print("Telegram startup message sent", flush=True)
        except Exception as exc:
            print(
                f"Telegram startup message failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
    while True:
        cycle_started = time.monotonic()
        try:
            symbols = run_scan(data_dir, top, ttl_hours)[:max_book_symbols]
            print(f"Scan complete; active order-book candidates: {symbols or 'none'}", flush=True)
        except Exception as exc:
            print(f"Scan failed: {type(exc).__name__}: {exc}", flush=True)
            symbols = []

        remaining = max(5.0, interval_seconds - (time.monotonic() - cycle_started))
        if symbols:
            output = data_dir / "order_book" / f"{_utc_now():%Y-%m-%d}.jsonl.gz"
            try:
                asyncio.run(
                    collect(
                        symbols,
                        output,
                        duration_seconds=remaining,
                        sample_interval_seconds=sample_seconds,
                    )
                )
            except Exception as exc:
                print(f"Order-book stream failed: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(60, remaining))
        else:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
