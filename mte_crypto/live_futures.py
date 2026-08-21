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

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from .live_spot import BinanceApiError


UTC = timezone.utc
STATE_FILENAME = "live_futures.json"
STATUS_FILENAME = "live_futures_connection.json"
LIVE_CONFIRMATION = "ENABLE_MTE_REAL_FUTURES_2X"


class BinanceFuturesPrivateClient:
    """Minimal RSA Binance USD-M Futures client; secrets are never persisted."""

    def __init__(
        self,
        api_key: str,
        private_key_pem: str | bytes,
        *,
        base_url: str = "https://fapi.binance.com",
        timeout: float = 15.0,
    ):
        self.api_key = api_key.strip()
        key_bytes = (
            private_key_pem.encode("utf-8")
            if isinstance(private_key_pem, str)
            else private_key_pem
        )
        private_key = load_pem_private_key(key_bytes, password=None)
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("Binance USD-M Futures requires an RSA private key")
        self.private_key = private_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._time_offset_ms = 0

    @classmethod
    def from_environment(cls) -> BinanceFuturesPrivateClient | None:
        api_key = os.environ.get("BINANCE_FUTURES_API_KEY", "").strip()
        raw_key = os.environ.get("BINANCE_FUTURES_RSA_PRIVATE_KEY", "").strip()
        encoded_key = os.environ.get(
            "BINANCE_FUTURES_RSA_PRIVATE_KEY_B64", ""
        ).strip()
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
            signature = self.private_key.sign(
                payload.encode("ASCII"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            values["signature"] = base64.b64encode(signature).decode("ASCII")
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
                "User-Agent": "MTE-Crypto-Hunter/0.3",
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
            raise BinanceApiError(f"Binance Futures connection failed: {exc}") from exc

    def sync_time(self) -> int:
        result = self._request("GET", "/fapi/v1/time")
        server_time = int(result["serverTime"])
        self._time_offset_ms = server_time - int(time.time() * 1000)
        return self._time_offset_ms

    def account_config(self) -> dict:
        return self._request("GET", "/fapi/v1/accountConfig", signed=True)

    def account_information(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def balances(self) -> list[dict]:
        return self._request("GET", "/fapi/v3/balance", signed=True)

    def exchange_info(self, symbol: str) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def positions(self, symbol: str | None = None) -> list[dict]:
        return self._request(
            "GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True
        )

    def change_margin_type(self, symbol: str, margin_type: str) -> dict:
        return self._request(
            "POST",
            "/fapi/v1/marginType",
            {"symbol": symbol, "marginType": margin_type},
            signed=True,
        )

    def change_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
            signed=True,
        )

    def new_order(self, **params) -> dict:
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def query_order(
        self,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "orderId": order_id,
                "origClientOrderId": client_order_id,
            },
            signed=True,
        )

    def new_algo_order(self, **params) -> dict:
        return self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

    def query_algo_order(
        self,
        *,
        algo_id: int | None = None,
        client_algo_id: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            "/fapi/v1/algoOrder",
            {"algoId": algo_id, "clientAlgoId": client_algo_id},
            signed=True,
        )

    def cancel_algo_order(
        self,
        *,
        algo_id: int | None = None,
        client_algo_id: str | None = None,
    ) -> dict:
        return self._request(
            "DELETE",
            "/fapi/v1/algoOrder",
            {"algoId": algo_id, "clientAlgoId": client_algo_id},
            signed=True,
        )


@dataclass(frozen=True)
class LiveFuturesConfig:
    enabled: bool = False
    confirmation: str = ""
    max_positions: int = 8
    margin_usdt: float = 11.0
    reserve_usdt: float = 12.0
    daily_loss_limit_usdt: float = 8.0
    leverage: int = 2
    initial_stop_pct: float = 0.075
    taker_fee_rate: float = 0.0005

    @property
    def armed(self) -> bool:
        return self.enabled and self.confirmation == LIVE_CONFIRMATION

    @classmethod
    def from_environment(cls) -> LiveFuturesConfig:
        return cls(
            enabled=os.environ.get("MTE_LIVE_FUTURES_ENABLED", "").lower()
            in {"1", "true", "yes"},
            confirmation=os.environ.get("MTE_LIVE_FUTURES_CONFIRMATION", ""),
            max_positions=max(
                1, min(16, int(os.environ.get("MTE_LIVE_FUTURES_MAX_POSITIONS", "8")))
            ),
            margin_usdt=max(
                5.0, float(os.environ.get("MTE_LIVE_FUTURES_MARGIN_USDT", "11"))
            ),
            reserve_usdt=max(
                0.0, float(os.environ.get("MTE_LIVE_FUTURES_RESERVE_USDT", "12"))
            ),
            daily_loss_limit_usdt=max(
                1.0,
                float(
                    os.environ.get("MTE_LIVE_FUTURES_DAILY_LOSS_LIMIT_USDT", "8")
                ),
            ),
            leverage=2,
            initial_stop_pct=min(
                0.20,
                max(
                    0.01,
                    float(os.environ.get("MTE_LIVE_FUTURES_INITIAL_STOP_PCT", "0.075")),
                ),
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


def _safe_number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _append(items: list, item: dict, limit: int = 500) -> None:
    items.append(item)
    if len(items) > limit:
        del items[:-limit]


def _new_state(now: datetime, cfg: LiveFuturesConfig) -> dict:
    return {
        "version": 1,
        "mode": "LIVE_FUTURES_2X_ARMED" if cfg.armed else "SAFE_DISABLED",
        "created_at": _timestamp(now),
        "updated_at": _timestamp(now),
        "config": {
            key: value for key, value in asdict(cfg).items() if key != "confirmation"
        },
        "positions": {},
        "closed_trades": [],
        "events": [],
    }


def ensure_live_futures_state(
    data_dir: Path, now: datetime, cfg: LiveFuturesConfig
) -> dict:
    path = data_dir / STATE_FILENAME
    state = _read_json(path)
    if not state.get("version"):
        state = _new_state(now, cfg)
    if cfg.armed:
        state["mode"] = "LIVE_FUTURES_2X_ARMED"
    elif state.get("positions"):
        state["mode"] = "MANAGE_ONLY"
    else:
        state["mode"] = "SAFE_DISABLED"
    state["config"] = {
        key: value for key, value in asdict(cfg).items() if key != "confirmation"
    }
    return state


def _usdt_available(balances: list[dict]) -> float:
    for balance in balances:
        if balance.get("asset") == "USDT":
            return _safe_number(balance.get("availableBalance"))
    return 0.0


def refresh_futures_connection_status(
    data_dir: Path,
    *,
    now: datetime,
    cfg: LiveFuturesConfig | None = None,
    client: BinanceFuturesPrivateClient | None = None,
) -> dict:
    cfg = cfg or LiveFuturesConfig.from_environment()
    api_present = bool(os.environ.get("BINANCE_FUTURES_API_KEY", "").strip())
    key_present = bool(
        os.environ.get("BINANCE_FUTURES_RSA_PRIVATE_KEY", "").strip()
        or os.environ.get("BINANCE_FUTURES_RSA_PRIVATE_KEY_B64", "").strip()
    )
    status = {
        "observed_at": _timestamp(now),
        "connected": False,
        "api_key_present": api_present,
        "private_key_present": key_present,
        "live_enabled": cfg.enabled,
        "live_armed": cfg.armed,
        "mode": "LIVE_FUTURES_2X_ARMED" if cfg.armed else "SAFE_DISABLED",
        "leverage": cfg.leverage,
    }
    if not api_present or not key_present:
        status["reason"] = "CREDENTIALS_INCOMPLETE"
    else:
        try:
            client = client or BinanceFuturesPrivateClient.from_environment()
            if client is None:
                raise BinanceApiError("Credentials are incomplete")
            client.sync_time()
            account = client.account_information()
            account_config = client.account_config()
            balances = client.balances()
            status.update(
                {
                    "connected": True,
                    "reason": "OK",
                    "can_trade": bool(account.get("canTrade")),
                    "can_withdraw": bool(account.get("canWithdraw")),
                    "can_deposit": bool(account.get("canDeposit")),
                    "dual_side_position": bool(
                        account_config.get("dualSidePosition")
                    ),
                    "multi_assets_margin": bool(
                        account_config.get("multiAssetsMargin")
                    ),
                    "usdt_available": _usdt_available(balances),
                }
            )
        except Exception as exc:
            status["reason"] = f"{type(exc).__name__}: {exc}"
    _write_json(data_dir / STATUS_FILENAME, status)
    return status


def _symbol_meta(client: BinanceFuturesPrivateClient, symbol: str) -> dict:
    response = client.exchange_info(symbol)
    symbols = response.get("symbols") or []
    meta = next((item for item in symbols if item.get("symbol") == symbol), None)
    if meta is None:
        raise BinanceApiError(f"No USD-M Futures exchange info for {symbol}")
    return meta


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


def _client_id(prefix: str, symbol: str, now: datetime) -> str:
    milliseconds = int(now.timestamp() * 1000)
    return f"MTEF{prefix}{milliseconds}{symbol[:7]}"[:36]


def _confirmed_new_order(
    client: BinanceFuturesPrivateClient,
    *,
    symbol: str,
    client_order_id: str,
    **params,
) -> dict:
    try:
        return client.new_order(
            symbol=symbol,
            newClientOrderId=client_order_id,
            **params,
        )
    except BinanceApiError as original:
        if original.status not in {None, 503}:
            raise
        for delay in (0.25, 0.75, 1.5):
            time.sleep(delay)
            try:
                return client.query_order(symbol, client_order_id=client_order_id)
            except Exception:
                continue
        raise original


def _confirmed_new_algo_order(
    client: BinanceFuturesPrivateClient,
    *,
    client_algo_id: str,
    **params,
) -> dict:
    try:
        return client.new_algo_order(clientAlgoId=client_algo_id, **params)
    except BinanceApiError as original:
        if original.status not in {None, 503}:
            raise
        for delay in (0.25, 0.75, 1.5):
            time.sleep(delay)
            try:
                return client.query_algo_order(client_algo_id=client_algo_id)
            except Exception:
                continue
        raise original


def _current_long_position(
    client: BinanceFuturesPrivateClient, symbol: str
) -> dict:
    positions = client.positions(symbol)
    return next(
        (
            item
            for item in positions
            if item.get("symbol") == symbol
            and item.get("positionSide", "BOTH") == "BOTH"
        ),
        {},
    )


def _estimated_pnl(
    position: dict, exit_price: float, quantity: float, fee_rate: float
) -> float:
    entry = _safe_number(position.get("entry_price"))
    entry_fee = entry * quantity * fee_rate
    exit_fee = exit_price * quantity * fee_rate
    return quantity * (exit_price - entry) - entry_fee - exit_fee


def open_live_futures_position(
    data_dir: Path,
    paper_open: dict,
    snapshot: dict | None,
    *,
    now: datetime,
    cfg: LiveFuturesConfig,
    client: BinanceFuturesPrivateClient,
) -> dict:
    """Open a 2x isolated long and immediately attach an exchange hard stop."""
    state = ensure_live_futures_state(data_dir, now, cfg)
    symbol = str(paper_open.get("symbol") or "").upper()
    if not cfg.armed:
        return {"opened": False, "reason": "SAFE_DISABLED", "symbol": symbol}
    if not paper_open.get("opened") or not symbol:
        return {"opened": False, "reason": "INVALID_SIGNAL", "symbol": symbol}
    if not snapshot:
        return {"opened": False, "reason": "NO_USDM_PERPETUAL_QUOTE", "symbol": symbol}
    if symbol in state["positions"]:
        return {"opened": False, "reason": "DUPLICATE_SYMBOL", "symbol": symbol}
    if len(state["positions"]) >= cfg.max_positions:
        return {"opened": False, "reason": "NO_FREE_LIVE_SLOT", "symbol": symbol}
    if _daily_realized_loss(state, now) <= -cfg.daily_loss_limit_usdt:
        return {"opened": False, "reason": "DAILY_LOSS_KILL_SWITCH", "symbol": symbol}

    ask = _safe_number(snapshot.get("ask"))
    if ask <= 0:
        return {"opened": False, "reason": "INVALID_USDM_QUOTE", "symbol": symbol}
    client.sync_time()
    account = client.account_information()
    account_config = client.account_config()
    if not account.get("canTrade"):
        return {"opened": False, "reason": "BINANCE_TRADE_PERMISSION_OFF", "symbol": symbol}
    if account_config.get("dualSidePosition"):
        return {"opened": False, "reason": "HEDGE_MODE_UNSUPPORTED", "symbol": symbol}
    if account_config.get("multiAssetsMargin"):
        return {"opened": False, "reason": "MULTI_ASSET_MODE_UNSUPPORTED", "symbol": symbol}
    available = _usdt_available(client.balances())
    allocation = min(cfg.margin_usdt, max(0.0, available - cfg.reserve_usdt))
    if allocation < 5.0:
        return {"opened": False, "reason": "INSUFFICIENT_FUTURES_USDT", "symbol": symbol}

    meta = _symbol_meta(client, symbol)
    if (
        meta.get("status") != "TRADING"
        or meta.get("contractType") != "PERPETUAL"
        or meta.get("quoteAsset") != "USDT"
    ):
        return {"opened": False, "reason": "UNSUPPORTED_USDM_CONTRACT", "symbol": symbol}
    lot = _filter(meta, "MARKET_LOT_SIZE") or _filter(meta, "LOT_SIZE")
    price_filter = _filter(meta, "PRICE_FILTER")
    notional_filter = _filter(meta, "MIN_NOTIONAL")
    notional = allocation * cfg.leverage
    min_notional = _safe_number(
        notional_filter.get("notional") or notional_filter.get("minNotional")
    ) or 5.0
    if notional < min_notional:
        return {"opened": False, "reason": "BELOW_SYMBOL_MIN_NOTIONAL", "symbol": symbol}
    quantity_text = _round_down(notional / ask, lot.get("stepSize") or "0.00000001")
    quantity = _safe_number(quantity_text)
    min_qty = _safe_number(lot.get("minQty"))
    max_qty = _safe_number(lot.get("maxQty"))
    if not quantity or quantity < min_qty or (max_qty and quantity > max_qty):
        return {"opened": False, "reason": "INVALID_FUTURES_QUANTITY", "symbol": symbol}

    try:
        client.change_margin_type(symbol, "ISOLATED")
    except BinanceApiError as exc:
        if exc.code != -4046:  # "No need to change margin type."
            raise
    leverage = client.change_leverage(symbol, cfg.leverage)
    if int(leverage.get("leverage") or 0) != cfg.leverage:
        raise BinanceApiError("Binance did not confirm 2x leverage")

    entry_client_id = _client_id("B", symbol, now)
    entry = _confirmed_new_order(
        client,
        symbol=symbol,
        client_order_id=entry_client_id,
        side="BUY",
        type="MARKET",
        quantity=quantity_text,
        newOrderRespType="RESULT",
    )
    executed_qty = _safe_number(entry.get("executedQty"))
    if executed_qty <= 0:
        raise BinanceApiError("Binance did not confirm a filled Futures entry")
    executed_text = _round_down(executed_qty, lot.get("stepSize") or "0.00000001")
    executed_qty = _safe_number(executed_text)
    entry_price = _safe_number(entry.get("avgPrice"))
    if not entry_price:
        cum_quote = _safe_number(entry.get("cumQuote"))
        entry_price = cum_quote / executed_qty if executed_qty and cum_quote else ask
    stop_price = entry_price * (1.0 - cfg.initial_stop_pct)
    stop_text = _round_down(stop_price, price_filter.get("tickSize") or "0.00000001")

    provisional = {
        "symbol": symbol,
        "alert_id": paper_open.get("alert_id"),
        "paper_slot_id": paper_open.get("slot_id"),
        "opened_at": _timestamp(now),
        "entry_order_id": entry.get("orderId"),
        "entry_client_order_id": entry_client_id,
        "quantity": executed_qty,
        "quantity_text": executed_text,
        "entry_price": entry_price,
        "entry_margin": allocation,
        "entry_notional": entry_price * executed_qty,
        "leverage": cfg.leverage,
        "margin_type": "ISOLATED",
        "hard_stop_price": _safe_number(stop_text),
        "status": "UNPROTECTED_PENDING_STOP",
        "updated_at": _timestamp(now),
    }
    state["positions"][symbol] = provisional
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)

    try:
        stop_client_id = _client_id("S", symbol, now)
        stop = _confirmed_new_algo_order(
            client,
            client_algo_id=stop_client_id,
            algoType="CONDITIONAL",
            symbol=symbol,
            side="SELL",
            type="STOP_MARKET",
            quantity=executed_text,
            triggerPrice=stop_text,
            workingType="MARK_PRICE",
            reduceOnly="true",
            priceProtect="false",
        )
    except Exception:
        try:
            _confirmed_new_order(
                client,
                symbol=symbol,
                client_order_id=_client_id("F", symbol, now),
                side="SELL",
                type="MARKET",
                quantity=executed_text,
                reduceOnly="true",
                newOrderRespType="RESULT",
            )
        except Exception:
            # Keep the provisional position in durable state so the next cycle
            # continues risk reconciliation if the emergency close is unknown.
            raise
        del state["positions"][symbol]
        _append(
            state["events"],
            {
                "observed_at": _timestamp(now),
                "type": "EMERGENCY_FLATTEN_AFTER_STOP_FAILURE",
                "symbol": symbol,
            },
        )
        state["updated_at"] = _timestamp(now)
        _write_json(data_dir / STATE_FILENAME, state)
        raise

    position = {
        **provisional,
        "stop_algo_id": stop.get("algoId"),
        "stop_client_algo_id": stop_client_id,
        "status": "OPEN_PROTECTED",
        "updated_at": _timestamp(now),
    }
    state["positions"][symbol] = position
    _append(state["events"], {"observed_at": _timestamp(now), "type": "OPEN", **position})
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return {"opened": True, **position}


def close_live_futures_positions(
    data_dir: Path,
    paper_closed: list[dict],
    *,
    now: datetime,
    cfg: LiveFuturesConfig,
    client: BinanceFuturesPrivateClient,
) -> list[dict]:
    """Reconcile server stops and mirror the Wave Rider paper exits."""
    state = ensure_live_futures_state(data_dir, now, cfg)
    closed: list[dict] = []

    for symbol, position in list(state["positions"].items()):
        current = _current_long_position(client, symbol)
        current_quantity = _safe_number(current.get("positionAmt"))
        if (
            current_quantity > 0
            and position.get("status") == "UNPROTECTED_PENDING_STOP"
        ):
            lot = _filter(_symbol_meta(client, symbol), "MARKET_LOT_SIZE")
            quantity_text = _round_down(
                current_quantity,
                lot.get("stepSize")
                or position.get("quantity_text")
                or "0.00000001",
            )
            exit_order = _confirmed_new_order(
                client,
                symbol=symbol,
                client_order_id=_client_id("R", symbol, now),
                side="SELL",
                type="MARKET",
                quantity=quantity_text,
                reduceOnly="true",
                newOrderRespType="RESULT",
            )
            executed = _safe_number(exit_order.get("executedQty")) or current_quantity
            exit_price = _safe_number(exit_order.get("avgPrice"))
            if not exit_price:
                cum_quote = _safe_number(exit_order.get("cumQuote"))
                exit_price = (
                    cum_quote / executed
                    if executed and cum_quote
                    else _safe_number(position.get("entry_price"))
                )
            trade = {
                **position,
                "closed_at": _timestamp(now),
                "exit_order_id": exit_order.get("orderId"),
                "exit_price": exit_price,
                "pnl": _estimated_pnl(
                    position, exit_price, executed, cfg.taker_fee_rate
                ),
                "exit_reason": "EMERGENCY_UNPROTECTED_FLATTEN",
            }
            closed.append(trade)
            _append(state["closed_trades"], trade)
            del state["positions"][symbol]
            continue
        if current_quantity > 0:
            continue
        algo = {}
        try:
            algo_id = int(position.get("stop_algo_id") or 0) or None
            algo = client.query_algo_order(
                algo_id=algo_id,
                client_algo_id=None if algo_id else position.get("stop_client_algo_id"),
            )
        except BinanceApiError:
            pass
        exit_price = _safe_number(algo.get("actualPrice")) or _safe_number(
            position.get("hard_stop_price")
        )
        quantity = _safe_number(algo.get("actualQty")) or _safe_number(
            position.get("quantity")
        )
        stopped = bool(algo.get("actualOrderId")) or _safe_number(algo.get("actualQty")) > 0
        trade = {
            **position,
            "closed_at": _timestamp(now),
            "exit_order_id": algo.get("actualOrderId"),
            "exit_price": exit_price,
            "pnl": _estimated_pnl(position, exit_price, quantity, cfg.taker_fee_rate),
            "exit_reason": "EXCHANGE_HARD_STOP" if stopped else "EXTERNAL_OR_MANUAL_CLOSE",
        }
        closed.append(trade)
        _append(state["closed_trades"], trade)
        del state["positions"][symbol]

    for paper_trade in paper_closed:
        symbol = str(paper_trade.get("symbol") or "").upper()
        position = state["positions"].get(symbol)
        if not position:
            continue
        current = _current_long_position(client, symbol)
        quantity = _safe_number(current.get("positionAmt"))
        if quantity <= 0:
            continue
        try:
            algo_id = int(position.get("stop_algo_id") or 0) or None
            client.cancel_algo_order(
                algo_id=algo_id,
                client_algo_id=None if algo_id else position.get("stop_client_algo_id"),
            )
        except BinanceApiError:
            # Risk reduction must continue. Any surviving algo is reduce-only.
            pass
        lot = _filter(_symbol_meta(client, symbol), "MARKET_LOT_SIZE")
        quantity_text = _round_down(
            quantity, lot.get("stepSize") or position.get("quantity_text") or "0.00000001"
        )
        exit_order = _confirmed_new_order(
            client,
            symbol=symbol,
            client_order_id=_client_id("X", symbol, now),
            side="SELL",
            type="MARKET",
            quantity=quantity_text,
            reduceOnly="true",
            newOrderRespType="RESULT",
        )
        executed = _safe_number(exit_order.get("executedQty")) or quantity
        exit_price = _safe_number(exit_order.get("avgPrice"))
        if not exit_price:
            cum_quote = _safe_number(exit_order.get("cumQuote"))
            exit_price = cum_quote / executed if executed and cum_quote else _safe_number(
                paper_trade.get("exit_price")
            )
        trade = {
            **position,
            "closed_at": _timestamp(now),
            "exit_order_id": exit_order.get("orderId"),
            "exit_price": exit_price,
            "pnl": _estimated_pnl(position, exit_price, executed, cfg.taker_fee_rate),
            "exit_reason": str(paper_trade.get("exit_reason") or "PAPER_WAVE_RIDER_EXIT"),
        }
        closed.append(trade)
        _append(state["closed_trades"], trade)
        del state["positions"][symbol]

    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return closed


def live_futures_payload(data_dir: Path) -> dict:
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
            "dual_side_position",
            "multi_assets_margin",
            "leverage",
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
