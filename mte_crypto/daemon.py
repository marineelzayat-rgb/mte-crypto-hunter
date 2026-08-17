from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from .alert_store import bootstrap_active_alerts, record_alert, update_alert_outcomes
from .binance import (
    BinanceFuturesPublicClient,
    BinancePublicClient,
    spot_usdt_tickers,
    usd_m_futures_snapshots,
)
from .book_collector import collect
from .config import DEFAULT_CONFIG
from .features import wilder_atr
from .futures_shadow import (
    ensure_futures_shadow,
    futures_shadow_payload,
    open_futures_shadow,
    update_futures_shadow,
)
from .monitor import format_alert, send_telegram
from .paper_portfolio import (
    ensure_paper_portfolio,
    open_paper_position,
    paper_portfolio_payload,
    update_paper_portfolio,
)
from .pulse import (
    DEFAULT_PULSE_CONFIG,
    PulseConfig,
    append_history,
    evaluate_pulses,
    format_pulse_alert,
    update_pulse_candidates,
)
from .scan import scan_market
from .status_server import start_status_server


UTC = timezone.utc
CAIRO = ZoneInfo("Africa/Cairo")
ALERT_STATE = "HUNTER_ALERT"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _percent(value) -> str:
    return f"{100 * float(value or 0):+.2f}%"


def _money(value) -> str:
    return f"${float(value or 0):,.2f}"


def send_paper_telegram_updates(
    data_dir: Path,
    spot_closed: list[dict],
    futures_closed: list[dict],
    *,
    now: datetime,
) -> list[str]:
    """Send close notifications and one 21:00 Cairo paper summary per day."""
    if not (
        os.environ.get("MTE_TELEGRAM_BOT_TOKEN")
        and os.environ.get("MTE_TELEGRAM_CHAT_ID")
    ):
        return []
    spot = paper_portfolio_payload(data_dir)
    futures = futures_shadow_payload(data_dir)
    futures_by_slot = {int(item["slot_id"]): item for item in futures_closed}
    sent_messages: list[str] = []

    for trade in spot_closed:
        shadow = futures_by_slot.get(int(trade["slot_id"]))
        futures_line = (
            f"Futures 2x: {_percent(shadow.get('return'))} "
            f"({_money(shadow.get('pnl'))})"
            if shadow
            else "Futures 2x: not available for this trade"
        )
        message = (
            "MTE PAPER — TRADE CLOSED\n"
            f"{trade.get('symbol')} | {trade.get('exit_reason')}\n"
            f"Spot 1x: {_percent(trade.get('return'))} "
            f"({_money(trade.get('pnl'))})\n"
            f"{futures_line}\n"
            f"Balances: Spot {_money(spot.get('equity'))} | "
            f"Futures 2x {_money(futures.get('equity'))}\n"
            "Paper research only — no real order"
        )
        try:
            if send_telegram(message):
                sent_messages.append(message)
        except Exception as exc:
            print(
                f"Paper close Telegram failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    local_now = now.astimezone(CAIRO)
    summary_path = data_dir / "telegram_paper_summary.json"
    summary_state = _read_json(summary_path)
    today = local_now.date().isoformat()
    if local_now.hour >= 21 and summary_state.get("last_sent_date") != today:
        message = (
            "MTE PAPER — DAILY SUMMARY\n"
            f"Spot 1x balance: {_money(spot.get('equity'))} "
            f"({_percent(spot.get('total_return'))})\n"
            f"Futures 2x balance: {_money(futures.get('equity'))} "
            f"({_percent(futures.get('total_return'))})\n"
            f"Open positions: Spot {spot.get('open_count', 0)}/16 | "
            f"Futures {futures.get('open_count', 0)}/16\n"
            f"Closed trades: Spot {len(spot.get('closed_trades') or [])} | "
            f"Futures {len(futures.get('closed_trades') or [])}\n"
            "Paper research only — no real orders"
        )
        try:
            if send_telegram(message):
                sent_messages.append(message)
                _write_json(
                    summary_path,
                    {"last_sent_date": today, "sent_at": now.isoformat()},
                )
        except Exception as exc:
            print(
                f"Daily paper Telegram failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
    return sent_messages


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
        record_alert(data_dir, row, source="hunter", observed_at=now)
        message = format_alert(row)
        if not send_telegram(message):
            print(message, flush=True)
    return sorted(active)


def run_pulse(
    data_dir: Path,
    cfg: PulseConfig,
    *,
    muted_alert_symbols: set[str] | None = None,
) -> list[str]:
    now = _utc_now()
    client = BinancePublicClient(DEFAULT_CONFIG.api_base)
    markets = spot_usdt_tickers(client)
    futures_snapshots: dict[str, dict] = {}
    try:
        futures_snapshots = usd_m_futures_snapshots(BinanceFuturesPublicClient())
    except Exception as exc:
        print(
            f"Futures shadow market data failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    history_path = data_dir / "pulse_history.json"
    active_path = data_dir / "pulse_candidates.json"
    history = _read_json(history_path)
    rows = evaluate_pulses(markets, history, now=now, cfg=cfg)
    next_history = append_history(
        markets,
        history,
        now=now,
        history_minutes=cfg.history_minutes,
    )
    active, new_alerts = update_pulse_candidates(
        rows,
        _read_json(active_path),
        now=now,
        ttl_minutes=cfg.ttl_minutes,
    )

    _write_json(history_path, next_history)
    _write_json(active_path, active)
    _write_json(
        data_dir / "latest_pulse.json",
        {"observed_at": now.isoformat(), "candidates": rows},
    )
    muted = muted_alert_symbols or set()
    for row in new_alerts:
        record = record_alert(data_dir, row, source="pulse", observed_at=now)
        spot_open = open_paper_position(data_dir, record, now=now)
        if spot_open.get("opened"):
            open_futures_shadow(
                data_dir,
                spot_open,
                futures_snapshots.get(str(record.get("symbol") or "")),
                now=now,
            )
        if row["symbol"] in muted:
            continue
        message = format_pulse_alert(row)
        if not send_telegram(message):
            print(message, flush=True)
    update_alert_outcomes(data_dir, markets, now=now)

    def hourly_atr(symbol: str) -> tuple[float, str] | None:
        try:
            frame = client.klines(symbol, "1h", limit=40, closed_only=True)
        except Exception as exc:
            print(
                f"Paper ATR fetch failed for {symbol}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        if len(frame) < 14:
            return None
        atr = float(wilder_atr(frame, 14).iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return None
        return atr, frame.index[-1].isoformat()

    spot_closed = update_paper_portfolio(
        data_dir,
        markets,
        now=now,
        atr_provider=hourly_atr,
    )
    futures_closed = update_futures_shadow(
        data_dir,
        futures_snapshots,
        spot_closed,
        now=now,
    )
    send_paper_telegram_updates(
        data_dir,
        spot_closed,
        futures_closed,
        now=now,
    )

    priority = {"EARLY_PULSE": 0, "RAPID_MOVE_NO_CHASE": 1}
    return sorted(
        active,
        key=lambda symbol: (priority.get(active[symbol].get("state", ""), 9), symbol),
    )


def main() -> None:
    data_dir = Path(os.getenv("MTE_DATA_DIR", "output/live"))
    data_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_active_alerts(data_dir)
    ensure_paper_portfolio(data_dir)
    ensure_futures_shadow(data_dir)
    top = int(os.getenv("MTE_SCAN_TOP", "120"))
    interval_seconds = max(300, int(os.getenv("MTE_SCAN_INTERVAL_SECONDS", "3600")))
    ttl_hours = max(1, int(os.getenv("MTE_CANDIDATE_TTL_HOURS", "48")))
    max_book_symbols = max(1, int(os.getenv("MTE_MAX_BOOK_SYMBOLS", "10")))
    sample_seconds = max(0.25, float(os.getenv("MTE_BOOK_SAMPLE_SECONDS", "1")))
    pulse_interval_seconds = max(
        60, int(os.getenv("MTE_PULSE_INTERVAL_SECONDS", "300"))
    )
    pulse_cfg = replace(
        DEFAULT_PULSE_CONFIG,
        ttl_minutes=max(15, int(os.getenv("MTE_PULSE_TTL_MINUTES", "120"))),
    )

    port = int(os.getenv("PORT", "8080"))
    try:
        start_status_server(data_dir, port)
        print(f"Status server started on port {port}", flush=True)
    except OSError as exc:
        print(f"Status server failed: {type(exc).__name__}: {exc}", flush=True)

    print(
        f"MTE daemon started: top={top}, scan_every={interval_seconds}s, "
        f"pulse_every={pulse_interval_seconds}s, ttl={ttl_hours}h, data={data_dir}",
        flush=True,
    )
    if os.environ.get("MTE_TELEGRAM_BOT_TOKEN") and os.environ.get(
        "MTE_TELEGRAM_CHAT_ID"
    ):
        try:
            if send_telegram(
                "MTE Crypto Hunter is online. Scanning the top "
                f"{top} Binance Spot USDT altcoins every "
                f"{interval_seconds // 60} minutes, with an all-market "
                f"price/volume pulse every {pulse_interval_seconds // 60} minutes."
            ):
                print("Telegram startup message sent", flush=True)
        except Exception as exc:
            print(
                f"Telegram startup message failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
    next_full_scan = 0.0
    hunter_symbols: list[str] = []
    while True:
        cycle_started = time.monotonic()
        if cycle_started >= next_full_scan:
            try:
                hunter_symbols = run_scan(data_dir, top, ttl_hours)
                print(
                    f"Full scan complete; hunter candidates: {hunter_symbols or 'none'}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Scan failed: {type(exc).__name__}: {exc}", flush=True)
            next_full_scan = cycle_started + interval_seconds

        try:
            pulse_symbols = run_pulse(
                data_dir,
                pulse_cfg,
                muted_alert_symbols=set(hunter_symbols),
            )
            print(
                f"Pulse complete; pulse candidates: {pulse_symbols or 'none'}",
                flush=True,
            )
        except Exception as exc:
            print(f"Pulse failed: {type(exc).__name__}: {exc}", flush=True)
            pulse_symbols = []

        symbols = list(dict.fromkeys(hunter_symbols + pulse_symbols))[:max_book_symbols]
        print(f"Active order-book candidates: {symbols or 'none'}", flush=True)
        until_full_scan = max(5.0, next_full_scan - time.monotonic())
        remaining = max(5.0, min(float(pulse_interval_seconds), until_full_scan))
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
                time.sleep(remaining)
        else:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
