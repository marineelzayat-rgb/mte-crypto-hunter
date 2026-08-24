from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
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
from .market_regime import detect_bull_regime
from .live_spot import (
    BinanceSpotPrivateClient,
    LiveSpotConfig,
    close_live_positions,
    ensure_live_state,
    open_live_position,
    refresh_connection_status,
)
from .live_futures import (
    BinanceFuturesPrivateClient,
    LiveFuturesConfig,
    close_live_futures_positions,
    ensure_live_futures_state,
    open_live_futures_position,
    refresh_futures_connection_status,
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
HEARTBEAT_FILENAME = "daemon_heartbeat.json"


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


def watchdog_is_stale(last_beat: float, now: float, timeout_seconds: float) -> bool:
    return now - last_beat > timeout_seconds


class RuntimeWatchdog:
    """Persist runtime health and force Railway to restart a stuck worker."""

    def __init__(self, data_dir: Path, timeout_seconds: float = 600.0):
        self.path = data_dir / HEARTBEAT_FILENAME
        self.timeout_seconds = max(180.0, float(timeout_seconds))
        self._last_beat = time.monotonic()
        self._lock = threading.Lock()

    def beat(self, phase: str, **details) -> None:
        now = _utc_now()
        with self._lock:
            self._last_beat = time.monotonic()
        try:
            _write_json(
                self.path,
                {
                    "observed_at": now.isoformat(),
                    "status": "RUNNING",
                    "phase": phase,
                    "pid": os.getpid(),
                    "watchdog_timeout_seconds": self.timeout_seconds,
                    **details,
                },
            )
        except OSError as exc:
            # Runtime health reporting must never prevent the trading daemon
            # from binding its HTTP port or managing an existing position.
            print(
                f"Heartbeat write failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def start(self) -> None:
        thread = threading.Thread(
            target=self._monitor,
            name="mte-watchdog",
            daemon=True,
        )
        thread.start()

    def _monitor(self) -> None:
        check_every = min(30.0, self.timeout_seconds / 4.0)
        while True:
            time.sleep(check_every)
            now = time.monotonic()
            with self._lock:
                last_beat = self._last_beat
            if not watchdog_is_stale(last_beat, now, self.timeout_seconds):
                continue
            age = now - last_beat
            _write_json(
                self.path,
                {
                    "observed_at": _utc_now().isoformat(),
                    "status": "STALLED_RESTARTING",
                    "phase": "watchdog",
                    "pid": os.getpid(),
                    "age_seconds": age,
                    "watchdog_timeout_seconds": self.timeout_seconds,
                },
            )
            print(
                f"Runtime watchdog restarting stalled worker after {age:.1f}s",
                flush=True,
            )
            os._exit(70)


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


def send_live_telegram_updates(opened: list[dict], closed: list[dict]) -> list[str]:
    """Report real executions without exposing credentials or enabling trades."""
    if not (
        os.environ.get("MTE_TELEGRAM_BOT_TOKEN")
        and os.environ.get("MTE_TELEGRAM_CHAT_ID")
    ):
        return []
    messages: list[str] = []
    for position in opened:
        message = (
            "MTE LIVE SPOT — BUY FILLED\n"
            f"{position.get('symbol')} @ {_money(position.get('entry_price'))}\n"
            f"Amount: {_money(position.get('quote_spent'))}\n"
            f"Hard stop: {_money(position.get('hard_stop_price'))}\n"
            "Real Binance order — protected by exchange-side stop"
        )
        if send_telegram(message):
            messages.append(message)
    for trade in closed:
        message = (
            "MTE LIVE SPOT — POSITION CLOSED\n"
            f"{trade.get('symbol')} | {trade.get('exit_reason')}\n"
            f"Exit: {_money(trade.get('exit_price'))}\n"
            f"Realized P&L: {_money(trade.get('pnl'))}\n"
            "Real Binance execution"
        )
        if send_telegram(message):
            messages.append(message)
    return messages


def send_live_futures_telegram_updates(
    opened: list[dict], closed: list[dict]
) -> list[str]:
    """Report real Futures fills without exposing account details."""
    if not (
        os.environ.get("MTE_TELEGRAM_BOT_TOKEN")
        and os.environ.get("MTE_TELEGRAM_CHAT_ID")
    ):
        return []
    messages: list[str] = []
    for position in opened:
        message = (
            "MTE LIVE FUTURES 4x — LONG FILLED\n"
            f"{position.get('symbol')} @ {_money(position.get('entry_price'))}\n"
            f"Isolated margin: {_money(position.get('entry_margin'))}\n"
            f"Position notional: {_money(position.get('entry_notional'))}\n"
            f"Exchange hard stop: {_money(position.get('hard_stop_price'))}\n"
            "Real Binance USD-M order — isolated 4x compounding"
        )
        if send_telegram(message):
            messages.append(message)
    for trade in closed:
        message = (
            "MTE LIVE FUTURES 4x — POSITION CLOSED\n"
            f"{trade.get('symbol')} | {trade.get('exit_reason')}\n"
            f"Exit: {_money(trade.get('exit_price'))}\n"
            f"Estimated realized P&L: {_money(trade.get('pnl'))}\n"
            "Real Binance USD-M execution"
        )
        if send_telegram(message):
            messages.append(message)
    return messages


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


def build_order_book_collection_plan(
    data_dir: Path,
    hunter_symbols: list[str],
    pulse_symbols: list[str],
    *,
    now: datetime,
    max_symbols: int,
    base_sample_seconds: float,
) -> tuple[list[str], dict[str, float]]:
    """Prioritize every paper position and sample early signal flow most densely."""
    observed: dict[str, datetime] = {}
    paper_symbols: set[str] = set()

    def add(symbol: str, timestamp) -> None:
        symbol = str(symbol or "").upper()
        if not symbol:
            return
        try:
            parsed = datetime.fromisoformat(str(timestamp))
            if not parsed.tzinfo:
                parsed = parsed.replace(tzinfo=UTC)
            parsed = parsed.astimezone(UTC)
        except (TypeError, ValueError):
            parsed = now.astimezone(UTC)
        observed[symbol] = max(observed.get(symbol, parsed), parsed)

    paper = paper_portfolio_payload(data_dir)
    for position in paper.get("open_positions", []):
        symbol = str(position.get("symbol") or "").upper()
        if symbol:
            paper_symbols.add(symbol)
            add(symbol, position.get("opened_at"))

    active_sources = (
        (hunter_symbols, _read_json(data_dir / "active_candidates.json")),
        (pulse_symbols, _read_json(data_dir / "pulse_candidates.json")),
    )
    for symbols, records in active_sources:
        for symbol in symbols:
            item = records.get(symbol) or {}
            add(symbol, item.get("detected_at"))

    now_utc = now.astimezone(UTC)
    intervals: dict[str, float] = {}
    for symbol, detected_at in observed.items():
        age_minutes = max(0.0, (now_utc - detected_at).total_seconds() / 60.0)
        if age_minutes <= 30:
            interval = base_sample_seconds
        elif age_minutes <= 120:
            interval = base_sample_seconds * 5.0
        else:
            interval = base_sample_seconds * 15.0
        intervals[symbol] = max(0.25, interval)

    ordered = sorted(
        observed,
        key=lambda symbol: (
            0 if symbol in paper_symbols else 1,
            intervals[symbol],
            symbol,
        ),
    )[:max_symbols]
    return ordered, {symbol: intervals[symbol] for symbol in ordered}


def apply_scan_results(
    data_dir: Path,
    frame,
    *,
    now: datetime,
    ttl_hours: int,
) -> list[str]:
    """Persist a completed scan in the main thread to avoid file races."""
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


def run_scan(data_dir: Path, top: int, ttl_hours: int) -> list[str]:
    frame = scan_market(replace(DEFAULT_CONFIG, max_universe=top))
    return apply_scan_results(
        data_dir,
        frame,
        now=_utc_now(),
        ttl_hours=ttl_hours,
    )


def run_pulse(
    data_dir: Path,
    cfg: PulseConfig,
    *,
    muted_alert_symbols: set[str] | None = None,
) -> list[str]:
    now = _utc_now()
    client = BinancePublicClient(DEFAULT_CONFIG.api_base)
    markets = spot_usdt_tickers(client)
    regime = {
        "active": False,
        "state": "NORMAL",
        "observed_at": now.isoformat(),
        "reason": "BTC_REGIME_DATA_UNAVAILABLE",
    }
    try:
        btc_daily = client.klines("BTCUSDT", "1d", limit=30, closed_only=False)
        regime = detect_bull_regime(markets, btc_daily, now=now)
    except Exception as exc:
        print(
            f"Bull regime detection failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    bull_mode = bool(regime.get("active"))
    _write_json(data_dir / "market_regime.json", regime)
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
    rows = evaluate_pulses(
        markets,
        history,
        now=now,
        cfg=cfg,
        bull_mode=bull_mode,
    )
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
        {"observed_at": now.isoformat(), "market_regime": regime, "candidates": rows},
    )
    muted = muted_alert_symbols or set()
    new_records: dict[str, dict] = {}
    for row in new_alerts:
        record = record_alert(data_dir, row, source="pulse", observed_at=now)
        new_records[str(row["symbol"])] = record
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
        bull_mode=bull_mode,
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

    # Reconsider every still-valid entry candidate after stale positions and
    # stops have freed slots. This prevents a five-minute signal from being
    # permanently lost merely because all slots were occupied on its first tick.
    paper = paper_portfolio_payload(data_dir)
    open_symbols = {
        str(position.get("symbol") or "")
        for position in paper.get("open_positions", [])
    }
    available = int(paper.get("available_slots") or 0)
    spot_opened: list[dict] = []
    for row in rows:
        if available <= 0:
            break
        symbol = str(row.get("symbol") or "")
        if row.get("state") not in {"EARLY_PULSE", "BULL_CONTINUATION"}:
            continue
        if not symbol or symbol in open_symbols:
            continue
        record = new_records.get(symbol) or {
            **row,
            "id": (
                f"{(active.get(symbol) or {}).get('detected_at', now.isoformat())}"
                f"|{symbol}|{row.get('state')}"
            ),
        }
        spot_open = open_paper_position(data_dir, record, now=now)
        if not spot_open.get("opened"):
            continue
        spot_opened.append(spot_open)
        open_futures_shadow(
            data_dir,
            spot_open,
            futures_snapshots.get(symbol),
            now=now,
        )
        open_symbols.add(symbol)
        available -= 1

    futures_live_cfg = LiveFuturesConfig.from_environment()
    futures_live_client = None
    try:
        futures_live_client = BinanceFuturesPrivateClient.from_environment()
        refresh_futures_connection_status(
            data_dir,
            now=now,
            cfg=futures_live_cfg,
            client=futures_live_client,
        )
    except Exception as exc:
        print(
            f"Live Futures connection check failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    futures_live_opened: list[dict] = []
    futures_live_closed: list[dict] = []
    if futures_live_client is not None:
        try:
            futures_live_closed = close_live_futures_positions(
                data_dir,
                spot_closed,
                now=now,
                cfg=futures_live_cfg,
                client=futures_live_client,
                snapshots=futures_snapshots,
                atr_provider=hourly_atr,
                bull_mode=bull_mode,
            )
            if futures_live_cfg.armed:
                # Real Futures entries are driven by newly accepted alerts,
                # independently of the separate paper portfolio's cash/slots.
                # This also prevents replaying alerts that predate arming.
                for symbol, record in new_records.items():
                    if record.get("state") not in {
                        "EARLY_PULSE",
                        "BULL_CONTINUATION",
                    }:
                        continue
                    live_signal = {
                        "opened": True,
                        "symbol": symbol,
                        "alert_id": record.get("id"),
                        "signal_state": record.get("state"),
                        "price": record.get("price"),
                        "exit_driver": "LIVE_WAVE_RIDER",
                    }
                    result = open_live_futures_position(
                        data_dir,
                        live_signal,
                        futures_snapshots.get(symbol),
                        now=now,
                        cfg=futures_live_cfg,
                        client=futures_live_client,
                    )
                    if result.get("opened"):
                        futures_live_opened.append(result)
            send_live_futures_telegram_updates(
                futures_live_opened, futures_live_closed
            )
        except Exception as exc:
            print(
                f"Live Futures cycle failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    live_cfg = LiveSpotConfig.from_environment()
    live_client = None
    try:
        live_client = BinanceSpotPrivateClient.from_environment()
        refresh_connection_status(
            data_dir,
            now=now,
            cfg=live_cfg,
            client=live_client,
        )
    except Exception as exc:
        print(f"Live Spot connection check failed: {type(exc).__name__}: {exc}", flush=True)
    live_opened: list[dict] = []
    live_closed: list[dict] = []
    if live_client is not None:
        try:
            live_closed = close_live_positions(
                data_dir,
                spot_closed,
                now=now,
                cfg=live_cfg,
                client=live_client,
            )
            # Never open the same signal in both real Spot and real Futures.
            # Futures is the preferred real layer when both gates are armed.
            if live_cfg.armed and not futures_live_cfg.armed:
                for spot_open in spot_opened:
                    result = open_live_position(
                        data_dir,
                        spot_open,
                        now=now,
                        cfg=live_cfg,
                        client=live_client,
                    )
                    if result.get("opened"):
                        live_opened.append(result)
            send_live_telegram_updates(live_opened, live_closed)
        except Exception as exc:
            print(f"Live Spot cycle failed: {type(exc).__name__}: {exc}", flush=True)

    priority = {
        "EARLY_PULSE": 0,
        "BULL_CONTINUATION": 1,
        "RAPID_MOVE_NO_CHASE": 2,
    }
    return sorted(
        active,
        key=lambda symbol: (priority.get(active[symbol].get("state", ""), 9), symbol),
    )


def main() -> None:
    data_dir = Path(os.getenv("MTE_DATA_DIR", "output/live"))
    data_dir.mkdir(parents=True, exist_ok=True)

    # Bind the Railway HTTP port before touching persistent state or external
    # services.  This keeps the deployment observable while startup recovery
    # or a slow volume operation is in progress.
    port = int(os.getenv("PORT", "8080"))
    try:
        start_status_server(data_dir, port)
        print(f"Status server started on port {port}", flush=True)
    except OSError as exc:
        print(f"Status server failed: {type(exc).__name__}: {exc}", flush=True)

    watchdog = RuntimeWatchdog(
        data_dir,
        timeout_seconds=float(os.getenv("MTE_WATCHDOG_TIMEOUT_SECONDS", "600")),
    )
    watchdog.beat("startup_bootstrap_alerts")
    watchdog.start()
    bootstrap_active_alerts(data_dir)
    watchdog.beat("startup_paper_portfolio")
    ensure_paper_portfolio(data_dir)
    watchdog.beat("startup_futures_shadow")
    ensure_futures_shadow(data_dir)
    watchdog.beat("startup_live_spot")
    ensure_live_state(data_dir, _utc_now(), LiveSpotConfig.from_environment())
    watchdog.beat("startup_live_futures")
    ensure_live_futures_state(
        data_dir, _utc_now(), LiveFuturesConfig.from_environment()
    )
    top = int(os.getenv("MTE_SCAN_TOP", "120"))
    interval_seconds = max(300, int(os.getenv("MTE_SCAN_INTERVAL_SECONDS", "3600")))
    ttl_hours = max(1, int(os.getenv("MTE_CANDIDATE_TTL_HOURS", "48")))
    max_book_symbols = max(16, int(os.getenv("MTE_MAX_BOOK_SYMBOLS", "32")))
    sample_seconds = max(0.25, float(os.getenv("MTE_BOOK_SAMPLE_SECONDS", "1")))
    pulse_interval_seconds = max(
        60, int(os.getenv("MTE_PULSE_INTERVAL_SECONDS", "120"))
    )
    pulse_cfg = replace(
        DEFAULT_PULSE_CONFIG,
        ttl_minutes=max(15, int(os.getenv("MTE_PULSE_TTL_MINUTES", "120"))),
    )
    watchdog.beat("startup")

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
    hunter_symbols = sorted(_read_json(data_dir / "active_candidates.json"))
    scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mte-scan")
    scan_future: Future | None = None
    scan_started_at: str | None = None
    while True:
        cycle_started = time.monotonic()
        watchdog.beat(
            "cycle_start",
            full_scan_running=bool(scan_future and not scan_future.done()),
        )

        if scan_future is not None and scan_future.done():
            try:
                frame = scan_future.result()
                hunter_symbols = apply_scan_results(
                    data_dir,
                    frame,
                    now=_utc_now(),
                    ttl_hours=ttl_hours,
                )
                print(
                    f"Full scan complete; hunter candidates: {hunter_symbols or 'none'}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Scan failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                scan_future = None
                scan_started_at = None

        # The hourly 120-symbol scan is intentionally background work.  A slow
        # Binance kline response must never block the two-minute pulse, live
        # position management, or connection heartbeat.
        if cycle_started >= next_full_scan and scan_future is None:
            scan_started_at = _utc_now().isoformat()
            scan_future = scan_executor.submit(
                scan_market,
                replace(DEFAULT_CONFIG, max_universe=top),
            )
            next_full_scan = cycle_started + interval_seconds
            print("Full scan started in background", flush=True)

        watchdog.beat(
            "pulse",
            full_scan_running=bool(scan_future and not scan_future.done()),
            full_scan_started_at=scan_started_at,
        )
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

        symbols, book_intervals = build_order_book_collection_plan(
            data_dir,
            hunter_symbols,
            pulse_symbols,
            now=_utc_now(),
            max_symbols=max_book_symbols,
            base_sample_seconds=sample_seconds,
        )
        print(f"Active order-book candidates: {symbols or 'none'}", flush=True)
        until_full_scan = max(5.0, next_full_scan - time.monotonic())
        remaining = max(5.0, min(float(pulse_interval_seconds), until_full_scan))
        if symbols:
            output = data_dir / "order_book" / f"{_utc_now():%Y-%m-%d}.jsonl.gz"
            watchdog.beat(
                "order_book",
                symbols=len(symbols),
                deadline_seconds=remaining + 10.0,
                full_scan_running=bool(scan_future and not scan_future.done()),
            )
            try:
                asyncio.run(
                    collect(
                        symbols,
                        output,
                        duration_seconds=remaining,
                        sample_interval_seconds=sample_seconds,
                        symbol_sample_intervals=book_intervals,
                    )
                )
            except Exception as exc:
                print(f"Order-book stream failed: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(remaining)
        else:
            watchdog.beat("sleep", sleep_seconds=remaining)
            time.sleep(remaining)
        watchdog.beat(
            "cycle_complete",
            full_scan_running=bool(scan_future and not scan_future.done()),
        )


if __name__ == "__main__":
    main()
