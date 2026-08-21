from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import base64
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from cryptography.hazmat.primitives.serialization import load_pem_private_key


UTC = timezone.utc
STATE_FILENAME = "live_spot.json"
STATUS_FILENAME = "live_spot_connection.json"
LIVE_CONFIRMATION = "ENABLE_MTE_REAL_SPOT"


class BinanceApiError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code=None):
        super().__init__(message)
        self.status = status
        self.code = code


class BinanceSpotPrivateClient:
    """Minimal Ed25519 Binance Spot client; secrets never enter persisted state."""

    def __init__(
        self,
        api_key: str,
        private_key_pem: str | bytes,
        *,
        base_url: str = "https://api.binance.com",
        timeout: float = 15.0,
    ):
        self.api_key = api_key.strip()
        key_bytes = (
            private_key_pem.encode("utf-8")
            if isinstance(private_key_pem, str)
            else private_key_pem
        )
        self.private_key = load_pem_private_key(key_bytes, password=None)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._time_offset_ms = 0

    @classmethod
    def from_environment(cls) -> BinanceSpotPrivateClient | None:
        api_key = os.environ.get("BINANCE_API_KEY", "").strip()
        raw_key = os.environ.get("BINANCE_ED25519_PRIVATE_KEY", "").strip()
        encoded_key = os.environ.get("BINANCE_ED25519_PRIVATE_KEY_B64", "").strip()
        if not api_key or not (raw_key or encoded_key):
            return None
        if encoded_key:
            private_key = base64.b64decode(encoded_key).decode("utf-8")
        else:
            private_key = raw_key.replace("\\n", "\n")
        return cls(api_key, private_key)

    def _timestamp(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        *,
        signed: bool = False,
    ):
        values = {key: value for key, value in (params or {}).items() if value is not None}
        if signed:
            values.setdefault("recvWindow", 5000)
            values["timestamp"] = self._timestamp()
            payload = urllib.parse.urlencode(values, encoding="UTF-8")
            signature = base64.b64encode(
                self.private_key.sign(payload.encode("ASCII"))
            ).decode("ASCII")
            values["signature"] = signature
        body = urllib.parse.urlencode(values, encoding="UTF-8").encode("ASCII")
        url = f"{self.base_url}{path}"
        if method in {"GET", "DELETE"} and body:
            url = f"{url}?{body.decode('ASCII')}"
            body = None
        request = urllib.request.Request(
            url,
            data=body if method not in {"GET", "DELETE"} else None,
            method=method,
            headers={
                "X-MBX-APIKEY": self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "MTE-Crypto-Hunter/0.2",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                error = json.loads(raw)
            except json.JSONDecodeError:
                error = {"msg": raw or str(exc)}
            raise BinanceApiError(
                str(error.get("msg") or exc),
                status=exc.code,
                code=error.get("code"),
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BinanceApiError(f"Binance connection failed: {exc}") from exc

    def sync_time(self) -> int:
        result = self._request("GET", "/api/v3/time")
        server_time = int(result["serverTime"])
        self._time_offset_ms = server_time - int(time.time() * 1000)
        return self._time_offset_ms

    def account(self) -> dict:
        return self._request("GET", "/api/v3/account", signed=True)

    def exchange_info(self, symbol: str) -> dict:
        return self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})

    def new_order(self, **params) -> dict:
        return self._request("POST", "/api/v3/order", params, signed=True)

    def query_order(
        self,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            "/api/v3/order",
            {
                "symbol": symbol,
                "orderId": order_id,
                "origClientOrderId": client_order_id,
            },
            signed=True,
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request(
            "DELETE",
            "/api/v3/order",
            {"symbol": symbol, "orderId": order_id},
            signed=True,
        )


@dataclass(frozen=True)
class LiveSpotConfig:
    enabled: bool = False
    confirmation: str = ""
    max_positions: int = 8
    order_usdt: float = 11.0
    reserve_usdt: float = 12.0
    daily_loss_limit_usdt: float = 8.0
    initial_stop_pct: float = 0.075

    @property
    def armed(self) -> bool:
        return self.enabled and self.confirmation == LIVE_CONFIRMATION

    @classmethod
    def from_environment(cls) -> LiveSpotConfig:
        return cls(
            enabled=os.environ.get("MTE_LIVE_SPOT_ENABLED", "").lower()
            in {"1", "true", "yes"},
            confirmation=os.environ.get("MTE_LIVE_SPOT_CONFIRMATION", ""),
            max_positions=max(1, int(os.environ.get("MTE_LIVE_MAX_POSITIONS", "8"))),
            order_usdt=max(5.0, float(os.environ.get("MTE_LIVE_ORDER_USDT", "11"))),
            reserve_usdt=max(0.0, float(os.environ.get("MTE_LIVE_RESERVE_USDT", "12"))),
            daily_loss_limit_usdt=max(
                1.0, float(os.environ.get("MTE_LIVE_DAILY_LOSS_LIMIT_USDT", "8"))
            ),
            initial_stop_pct=min(
                0.20, max(0.01, float(os.environ.get("MTE_LIVE_INITIAL_STOP_PCT", "0.075")))
            ),
        )


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


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _new_state(now: datetime, cfg: LiveSpotConfig) -> dict:
    return {
        "version": 1,
        "mode": "LIVE_SPOT_ARMED" if cfg.armed else "SAFE_DISABLED",
        "created_at": _timestamp(now),
        "updated_at": _timestamp(now),
        "config": {
            key: value
            for key, value in asdict(cfg).items()
            if key != "confirmation"
        },
        "positions": {},
        "closed_trades": [],
        "events": [],
    }


def ensure_live_state(data_dir: Path, now: datetime, cfg: LiveSpotConfig) -> dict:
    path = data_dir / STATE_FILENAME
    state = _read_json(path)
    if not state.get("version"):
        state = _new_state(now, cfg)
    if cfg.armed:
        state["mode"] = "LIVE_SPOT_ARMED"
    elif state.get("positions"):
        state["mode"] = "MANAGE_ONLY"
    else:
        state["mode"] = "SAFE_DISABLED"
    state["config"] = {
        key: value for key, value in asdict(cfg).items() if key != "confirmation"
    }
    return state


def _append(items: list, item: dict, limit: int = 500) -> None:
    items.append(item)
    if len(items) > limit:
        del items[:-limit]


def _safe_number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _free_balance(account: dict, asset: str) -> float:
    for balance in account.get("balances", []):
        if balance.get("asset") == asset:
            return _safe_number(balance.get("free"))
    return 0.0


def refresh_connection_status(
    data_dir: Path,
    *,
    now: datetime,
    cfg: LiveSpotConfig | None = None,
    client: BinanceSpotPrivateClient | None = None,
) -> dict:
    cfg = cfg or LiveSpotConfig.from_environment()
    api_present = bool(os.environ.get("BINANCE_API_KEY", "").strip())
    key_present = bool(
        os.environ.get("BINANCE_ED25519_PRIVATE_KEY", "").strip()
        or os.environ.get("BINANCE_ED25519_PRIVATE_KEY_B64", "").strip()
    )
    status = {
        "observed_at": _timestamp(now),
        "connected": False,
        "api_key_present": api_present,
        "private_key_present": key_present,
        "live_enabled": cfg.enabled,
        "live_armed": cfg.armed,
        "mode": "SAFE_DISABLED" if not cfg.armed else "LIVE_SPOT_ARMED",
    }
    if not api_present or not key_present:
        status["reason"] = "CREDENTIALS_INCOMPLETE"
    else:
        try:
            client = client or BinanceSpotPrivateClient.from_environment()
            if client is None:
                raise BinanceApiError("Credentials are incomplete")
            client.sync_time()
            account = client.account()
            status.update(
                {
                    "connected": True,
                    "reason": "OK",
                    "can_trade": bool(account.get("canTrade")),
                    "can_withdraw": bool(account.get("canWithdraw")),
                    "can_deposit": bool(account.get("canDeposit")),
                    "usdt_free": _free_balance(account, "USDT"),
                    "account_type": account.get("accountType"),
                }
            )
        except Exception as exc:
            status["reason"] = f"{type(exc).__name__}: {exc}"
    _write_json(data_dir / STATUS_FILENAME, status)
    return status


def _symbol_meta(client: BinanceSpotPrivateClient, symbol: str) -> dict:
    response = client.exchange_info(symbol)
    symbols = response.get("symbols") or []
    if not symbols:
        raise BinanceApiError(f"No exchange info for {symbol}")
    return symbols[0]


def _filter(meta: dict, filter_type: str) -> dict:
    return next(
        (item for item in meta.get("filters", []) if item.get("filterType") == filter_type),
        {},
    )


def _round_down(value: float, increment: str | float) -> str:
    step = Decimal(str(increment))
    if step <= 0:
        return format(Decimal(str(value)), "f")
    units = (Decimal(str(value)) / step).to_integral_value(rounding=ROUND_DOWN)
    rounded = units * step
    return format(rounded.normalize(), "f")


def _daily_realized_loss(state: dict, now: datetime) -> float:
    today = now.astimezone(UTC).date()
    pnl = 0.0
    for trade in state.get("closed_trades", []):
        try:
            closed = datetime.fromisoformat(str(trade["closed_at"])).astimezone(UTC)
        except (KeyError, TypeError, ValueError):
            continue
        if closed.date() == today:
            pnl += _safe_number(trade.get("pnl"))
    return min(0.0, pnl)


def _client_order_id(prefix: str, symbol: str, now: datetime) -> str:
    milliseconds = int(now.timestamp() * 1000)
    return f"MTE{prefix}{milliseconds}{symbol[:8]}"[:36]


def _confirmed_new_order(
    client: BinanceSpotPrivateClient,
    *,
    symbol: str,
    client_order_id: str,
    **params,
) -> dict:
    """Resolve ambiguous network/timeouts by querying the idempotent client ID."""
    try:
        return client.new_order(
            symbol=symbol,
            newClientOrderId=client_order_id,
            **params,
        )
    except Exception as original:
        for delay in (0.25, 0.75, 1.5):
            time.sleep(delay)
            try:
                return client.query_order(symbol, client_order_id=client_order_id)
            except Exception:
                continue
        raise original


def open_live_position(
    data_dir: Path,
    paper_open: dict,
    *,
    now: datetime,
    cfg: LiveSpotConfig,
    client: BinanceSpotPrivateClient,
) -> dict:
    """Enter a newly accepted paper signal and immediately attach a hard stop."""
    state = ensure_live_state(data_dir, now, cfg)
    symbol = str(paper_open.get("symbol") or "").upper()
    if not cfg.armed:
        return {"opened": False, "reason": "SAFE_DISABLED", "symbol": symbol}
    if not paper_open.get("opened") or not symbol:
        return {"opened": False, "reason": "INVALID_SIGNAL", "symbol": symbol}
    if symbol in state["positions"]:
        return {"opened": False, "reason": "DUPLICATE_SYMBOL", "symbol": symbol}
    if len(state["positions"]) >= cfg.max_positions:
        return {"opened": False, "reason": "NO_FREE_LIVE_SLOT", "symbol": symbol}
    if _daily_realized_loss(state, now) <= -cfg.daily_loss_limit_usdt:
        return {"opened": False, "reason": "DAILY_LOSS_KILL_SWITCH", "symbol": symbol}

    client.sync_time()
    account = client.account()
    if not account.get("canTrade"):
        return {"opened": False, "reason": "BINANCE_TRADE_PERMISSION_OFF", "symbol": symbol}
    usdt_free = _free_balance(account, "USDT")
    spend = min(cfg.order_usdt, max(0.0, usdt_free - cfg.reserve_usdt))
    if spend < 5.0:
        return {"opened": False, "reason": "INSUFFICIENT_FREE_USDT", "symbol": symbol}

    meta = _symbol_meta(client, symbol)
    lot_size = _filter(meta, "LOT_SIZE")
    price_filter = _filter(meta, "PRICE_FILTER")
    notional_filter = _filter(meta, "NOTIONAL") or _filter(meta, "MIN_NOTIONAL")
    if "MARKET" not in (meta.get("orderTypes") or []):
        return {"opened": False, "reason": "MARKET_ORDER_UNSUPPORTED", "symbol": symbol}
    min_notional = _safe_number(notional_filter.get("minNotional")) or 5.0
    if spend < min_notional:
        return {"opened": False, "reason": "BELOW_SYMBOL_MIN_NOTIONAL", "symbol": symbol}

    buy_client_id = _client_order_id("B", symbol, now)
    buy = _confirmed_new_order(
        client,
        symbol=symbol,
        client_order_id=buy_client_id,
        side="BUY",
        type="MARKET",
        quoteOrderQty=f"{spend:.2f}",
        newOrderRespType="FULL",
    )
    executed_qty = _safe_number(buy.get("executedQty"))
    quote_spent = _safe_number(buy.get("cummulativeQuoteQty"))
    base_asset = str(meta.get("baseAsset") or "")
    base_commission = sum(
        _safe_number(fill.get("commission"))
        for fill in buy.get("fills", [])
        if fill.get("commissionAsset") == base_asset
    )
    quantity = max(0.0, executed_qty - base_commission)
    quantity_text = _round_down(quantity, lot_size.get("stepSize") or "0.00000001")
    quantity = _safe_number(quantity_text)
    entry_price = quote_spent / executed_qty if executed_qty else 0.0
    stop_price = entry_price * (1.0 - cfg.initial_stop_pct)
    stop_text = _round_down(stop_price, price_filter.get("tickSize") or "0.00000001")

    if not quantity or not entry_price or "STOP_LOSS" not in (meta.get("orderTypes") or []):
        if quantity:
            _confirmed_new_order(
                client,
                symbol=symbol,
                client_order_id=_client_order_id("F", symbol, now),
                side="SELL",
                type="MARKET",
                quantity=quantity_text,
            )
        event = {
            "observed_at": _timestamp(now),
            "symbol": symbol,
            "type": "EMERGENCY_FLATTEN",
            "reason": "PROTECTIVE_STOP_UNAVAILABLE",
        }
        _append(state["events"], event)
        state["updated_at"] = _timestamp(now)
        _write_json(data_dir / STATE_FILENAME, state)
        return {"opened": False, **event}

    try:
        stop_client_id = _client_order_id("S", symbol, now)
        stop = _confirmed_new_order(
            client,
            symbol=symbol,
            client_order_id=stop_client_id,
            side="SELL",
            type="STOP_LOSS",
            quantity=quantity_text,
            stopPrice=stop_text,
            newOrderRespType="RESULT",
        )
    except Exception:
        _confirmed_new_order(
            client,
            symbol=symbol,
            client_order_id=_client_order_id("F", symbol, now),
            side="SELL",
            type="MARKET",
            quantity=quantity_text,
        )
        raise

    position = {
        "symbol": symbol,
        "alert_id": paper_open.get("alert_id"),
        "paper_slot_id": paper_open.get("slot_id"),
        "opened_at": _timestamp(now),
        "buy_order_id": buy.get("orderId"),
        "buy_client_order_id": buy_client_id,
        "stop_order_id": stop.get("orderId"),
        "stop_client_order_id": stop_client_id,
        "quantity": quantity,
        "quantity_text": quantity_text,
        "entry_price": entry_price,
        "quote_spent": quote_spent,
        "hard_stop_price": _safe_number(stop_text),
        "status": "OPEN_PROTECTED",
        "updated_at": _timestamp(now),
    }
    state["positions"][symbol] = position
    _append(state["events"], {"observed_at": _timestamp(now), "type": "OPEN", **position})
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return {"opened": True, **position}


def close_live_positions(
    data_dir: Path,
    paper_closed: list[dict],
    *,
    now: datetime,
    cfg: LiveSpotConfig,
    client: BinanceSpotPrivateClient,
) -> list[dict]:
    """Mirror paper exits while keeping the exchange-side initial stop as backup."""
    state = ensure_live_state(data_dir, now, cfg)
    closed: list[dict] = []

    # A server-side stop may fill between scanner cycles or during a restart.
    # Reconcile every live position before reacting to paper exits.
    for symbol, position in list(state["positions"].items()):
        stop_id = int(position.get("stop_order_id") or 0)
        if not stop_id:
            continue
        stop_status = client.query_order(symbol, order_id=stop_id)
        if stop_status.get("status") != "FILLED":
            continue
        executed = _safe_number(stop_status.get("executedQty"))
        proceeds = _safe_number(stop_status.get("cummulativeQuoteQty"))
        exit_price = proceeds / executed if executed else _safe_number(
            position.get("hard_stop_price")
        )
        trade = {
            **position,
            "closed_at": _timestamp(now),
            "exit_order_id": stop_status.get("orderId"),
            "exit_price": exit_price,
            "proceeds": proceeds,
            "pnl": proceeds - _safe_number(position.get("quote_spent")),
            "exit_reason": "EXCHANGE_HARD_STOP",
        }
        closed.append(trade)
        _append(state["closed_trades"], trade)
        del state["positions"][symbol]

    for paper_trade in paper_closed:
        symbol = str(paper_trade.get("symbol") or "").upper()
        position = state["positions"].get(symbol)
        if not position:
            continue
        stop_id = int(position.get("stop_order_id") or 0)
        stop_status = client.query_order(symbol, order_id=stop_id) if stop_id else {}
        if stop_status.get("status") == "FILLED":
            sell = stop_status
            reason = "EXCHANGE_HARD_STOP"
        else:
            if stop_id:
                try:
                    client.cancel_order(symbol, stop_id)
                except BinanceApiError as exc:
                    if exc.code not in {-2011, -2013}:
                        raise
            sell = _confirmed_new_order(
                client,
                symbol=symbol,
                client_order_id=_client_order_id("X", symbol, now),
                side="SELL",
                type="MARKET",
                quantity=position.get("quantity_text") or str(position["quantity"]),
                newOrderRespType="FULL",
            )
            reason = str(paper_trade.get("exit_reason") or "PAPER_EXIT")
        executed = _safe_number(sell.get("executedQty"))
        proceeds = _safe_number(sell.get("cummulativeQuoteQty"))
        exit_price = proceeds / executed if executed else _safe_number(paper_trade.get("exit_price"))
        trade = {
            **position,
            "closed_at": _timestamp(now),
            "exit_order_id": sell.get("orderId"),
            "exit_price": exit_price,
            "proceeds": proceeds,
            "pnl": proceeds - _safe_number(position.get("quote_spent")),
            "exit_reason": reason,
        }
        closed.append(trade)
        _append(state["closed_trades"], trade)
        del state["positions"][symbol]
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return closed


def live_spot_payload(data_dir: Path) -> dict:
    state = _read_json(data_dir / STATE_FILENAME)
    connection = _read_json(data_dir / STATUS_FILENAME)
    positions = list((state.get("positions") or {}).values())
    safe_connection = {
        key: connection.get(key)
        for key in (
            "observed_at",
            "connected",
            "api_key_present",
            "private_key_present",
            "live_enabled",
            "live_armed",
            "mode",
            "reason",
            "can_trade",
            "account_type",
        )
        if key in connection
    }
    return {
        "mode": state.get("mode", "SAFE_DISABLED"),
        "connection": safe_connection,
        "config": state.get("config", {}),
        "open_count": len(positions),
        "closed_count": len(state.get("closed_trades") or []),
        "updated_at": state.get("updated_at"),
    }
