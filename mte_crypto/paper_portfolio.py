from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Callable


UTC = timezone.utc
STATE_FILENAME = "paper_portfolio.json"


@dataclass(frozen=True)
class PaperPortfolioConfig:
    starting_equity: float = 100.0
    max_positions: int = 16
    initial_stop_pct: float = 0.075
    activation_pct: float = 0.05
    atr_length: int = 14
    atr_multiple: float = 2.5
    max_hold_hours: int = 24
    fee_rate: float = 0.001


DEFAULT_PAPER_CONFIG = PaperPortfolioConfig()


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


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if not parsed.tzinfo else parsed.astimezone(UTC)


def _finite_positive(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _new_state(now: datetime, cfg: PaperPortfolioConfig) -> dict:
    slot_balance = cfg.starting_equity / cfg.max_positions
    return {
        "version": 1,
        "created_at": _timestamp(now),
        "updated_at": _timestamp(now),
        "config": asdict(cfg),
        "slots": [
            {"id": index + 1, "balance": slot_balance, "position": None}
            for index in range(cfg.max_positions)
        ],
        "closed_trades": [],
        "skipped_signals": [],
    }


def ensure_paper_portfolio(
    data_dir: Path,
    *,
    now: datetime | None = None,
    cfg: PaperPortfolioConfig = DEFAULT_PAPER_CONFIG,
) -> dict:
    path = data_dir / STATE_FILENAME
    state = _read_json(path)
    if state.get("slots"):
        return state
    state = _new_state(now or datetime.now(UTC), cfg)
    _write_json(path, state)
    return state


def _append_bounded(items: list, value: dict, *, limit: int = 500) -> None:
    items.append(value)
    if len(items) > limit:
        del items[:-limit]


def open_paper_position(
    data_dir: Path,
    alert: dict,
    *,
    now: datetime,
    cfg: PaperPortfolioConfig = DEFAULT_PAPER_CONFIG,
) -> dict:
    """Open one isolated paper slot for a new EARLY_PULSE alert."""
    state = ensure_paper_portfolio(data_dir, now=now, cfg=cfg)
    symbol = str(alert.get("symbol") or "")
    signal_state = str(alert.get("state") or "")
    price = _finite_positive(alert.get("price"))
    reason = None
    if signal_state != "EARLY_PULSE":
        reason = "NOT_EARLY_PULSE"
    elif not symbol or not price:
        reason = "INVALID_ALERT"
    elif any(
        (slot.get("position") or {}).get("symbol") == symbol
        for slot in state["slots"]
    ):
        reason = "DUPLICATE_SYMBOL"

    free_slot = next(
        (slot for slot in state["slots"] if not slot.get("position")),
        None,
    )
    if reason is None and free_slot is None:
        reason = "NO_FREE_SLOT"

    if reason is not None:
        skipped = {
            "observed_at": _timestamp(now),
            "symbol": symbol,
            "state": signal_state,
            "price": price or None,
            "reason": reason,
        }
        _append_bounded(state.setdefault("skipped_signals", []), skipped)
        state["updated_at"] = _timestamp(now)
        _write_json(data_dir / STATE_FILENAME, state)
        return {"opened": False, **skipped}

    allocation = float(free_slot["balance"])
    entry_fee = allocation * cfg.fee_rate
    quantity = (allocation - entry_fee) / price
    position = {
        "alert_id": alert.get("id"),
        "symbol": symbol,
        "opened_at": _timestamp(now),
        "entry_price": price,
        "entry_allocation": allocation,
        "entry_fee": entry_fee,
        "quantity": quantity,
        "current_price": price,
        "highest_price": price,
        "stop_price": price * (1.0 - cfg.initial_stop_pct),
        "trail_active": False,
        "last_atr": None,
        "last_trail_candle_at": None,
        "updated_at": _timestamp(now),
    }
    free_slot["balance"] = 0.0
    free_slot["position"] = position
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return {"opened": True, "slot_id": free_slot["id"], **position}


def _close_position(
    state: dict,
    slot: dict,
    *,
    exit_price: float,
    reason: str,
    now: datetime,
    cfg: PaperPortfolioConfig,
) -> dict:
    position = slot["position"]
    gross_proceeds = float(position["quantity"]) * exit_price
    exit_fee = gross_proceeds * cfg.fee_rate
    proceeds = gross_proceeds - exit_fee
    allocation = float(position["entry_allocation"])
    trade = {
        **position,
        "slot_id": slot["id"],
        "closed_at": _timestamp(now),
        "exit_price": exit_price,
        "exit_fee": exit_fee,
        "exit_reason": reason,
        "proceeds": proceeds,
        "pnl": proceeds - allocation,
        "return": proceeds / allocation - 1.0 if allocation else 0.0,
    }
    slot["balance"] = proceeds
    slot["position"] = None
    _append_bounded(state.setdefault("closed_trades", []), trade)
    return trade


AtrProvider = Callable[[str], tuple[float, str] | None]


def update_paper_portfolio(
    data_dir: Path,
    markets: list[dict],
    *,
    now: datetime,
    atr_provider: AtrProvider | None = None,
    cfg: PaperPortfolioConfig = DEFAULT_PAPER_CONFIG,
) -> list[dict]:
    """Mark positions, raise hourly Chandelier stops, and close paper trades."""
    state = ensure_paper_portfolio(data_dir, now=now, cfg=cfg)
    prices = {
        str(row.get("symbol") or ""): _finite_positive(row.get("last_price"))
        for row in markets
    }
    closed: list[dict] = []
    now_utc = now.astimezone(UTC)
    for slot in state["slots"]:
        position = slot.get("position")
        if not position:
            continue
        price = prices.get(str(position["symbol"]), 0.0)
        if not price:
            continue
        entry = float(position["entry_price"])
        position["current_price"] = price
        position["highest_price"] = max(float(position["highest_price"]), price)
        position["updated_at"] = _timestamp(now_utc)
        if position["highest_price"] >= entry * (1.0 + cfg.activation_pct):
            position["trail_active"] = True

        stop = float(position["stop_price"])
        if price <= stop:
            closed.append(
                _close_position(
                    state,
                    slot,
                    exit_price=price,
                    reason="TRAIL" if position.get("trail_active") else "INITIAL_STOP",
                    now=now_utc,
                    cfg=cfg,
                )
            )
            continue

        if position.get("trail_active") and atr_provider is not None:
            expected_candle = (
                now_utc.replace(minute=0, second=0, microsecond=0)
                - timedelta(hours=1)
            ).isoformat()
            snapshot = (
                atr_provider(str(position["symbol"]))
                if position.get("last_trail_candle_at") != expected_candle
                else None
            )
            if snapshot is not None:
                atr, candle_at = snapshot
                atr = _finite_positive(atr)
                if atr and candle_at != position.get("last_trail_candle_at"):
                    candidate = float(position["highest_price"]) - cfg.atr_multiple * atr
                    position["stop_price"] = max(stop, candidate)
                    position["last_atr"] = atr
                    position["last_trail_candle_at"] = candle_at
                    if price <= float(position["stop_price"]):
                        closed.append(
                            _close_position(
                                state,
                                slot,
                                exit_price=price,
                                reason="TRAIL",
                                now=now_utc,
                                cfg=cfg,
                            )
                        )
                        continue

        opened = _parse_timestamp(position["opened_at"])
        if now_utc - opened >= timedelta(hours=cfg.max_hold_hours):
            closed.append(
                _close_position(
                    state,
                    slot,
                    exit_price=price,
                    reason="TIME_24H",
                    now=now_utc,
                    cfg=cfg,
                )
            )

    state["updated_at"] = _timestamp(now_utc)
    _write_json(data_dir / STATE_FILENAME, state)
    return closed


def paper_portfolio_payload(
    data_dir: Path,
    *,
    cfg: PaperPortfolioConfig = DEFAULT_PAPER_CONFIG,
) -> dict:
    state = ensure_paper_portfolio(data_dir, cfg=cfg)
    open_positions = []
    cash = 0.0
    market_value = 0.0
    for slot in state["slots"]:
        position = slot.get("position")
        if not position:
            cash += float(slot.get("balance") or 0.0)
            continue
        current_price = _finite_positive(position.get("current_price"))
        value = float(position["quantity"]) * current_price
        market_value += value
        allocation = float(position["entry_allocation"])
        open_positions.append(
            {
                **position,
                "slot_id": slot["id"],
                "market_value": value,
                "unrealized_pnl": value - allocation,
                "current_return": value / allocation - 1.0 if allocation else 0.0,
            }
        )
    equity = cash + market_value
    closed = list(reversed(state.get("closed_trades", [])))
    return {
        "mode": "PAPER_ONLY",
        "starting_equity": cfg.starting_equity,
        "equity": equity,
        "total_return": equity / cfg.starting_equity - 1.0,
        "cash": cash,
        "market_value": market_value,
        "realized_pnl": sum(float(item.get("pnl") or 0.0) for item in closed),
        "open_count": len(open_positions),
        "available_slots": cfg.max_positions - len(open_positions),
        "max_positions": cfg.max_positions,
        "config": asdict(cfg),
        "open_positions": open_positions,
        "closed_trades": closed[:100],
        "skipped_signals": list(reversed(state.get("skipped_signals", [])))[:100],
        "updated_at": state.get("updated_at"),
    }
