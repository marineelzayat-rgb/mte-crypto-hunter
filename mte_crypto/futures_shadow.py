from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path


UTC = timezone.utc
STATE_FILENAME = "futures_shadow.json"
SAMPLES_FILENAME = "futures_shadow_samples.jsonl"


@dataclass(frozen=True)
class FuturesShadowConfig:
    starting_equity: float = 100.0
    max_positions: int = 16
    leverage: float = 2.0
    taker_fee_rate: float = 0.0005


DEFAULT_FUTURES_CONFIG = FuturesShadowConfig()


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _finite_positive(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


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


def _append_bounded(items: list, value: dict, *, limit: int = 500) -> None:
    items.append(value)
    if len(items) > limit:
        del items[:-limit]


def _new_state(now: datetime, cfg: FuturesShadowConfig) -> dict:
    slot_balance = cfg.starting_equity / cfg.max_positions
    return {
        "version": 1,
        "mode": "PAPER_ONLY_NO_ORDERS",
        "created_at": _timestamp(now),
        "updated_at": _timestamp(now),
        "config": asdict(cfg),
        "slots": [
            {"id": index + 1, "balance": slot_balance, "position": None}
            for index in range(cfg.max_positions)
        ],
        "closed_trades": [],
        "skipped_signals": [],
        "funding_events": [],
    }


def ensure_futures_shadow(
    data_dir: Path,
    *,
    now: datetime | None = None,
    cfg: FuturesShadowConfig = DEFAULT_FUTURES_CONFIG,
) -> dict:
    path = data_dir / STATE_FILENAME
    state = _read_json(path)
    if state.get("slots"):
        return state
    state = _new_state(now or datetime.now(UTC), cfg)
    _write_json(path, state)
    return state


def open_futures_shadow(
    data_dir: Path,
    spot_open: dict,
    snapshot: dict | None,
    *,
    now: datetime,
    cfg: FuturesShadowConfig = DEFAULT_FUTURES_CONFIG,
) -> dict:
    """Mirror an accepted spot paper entry at the USD-M perpetual ask."""
    state = ensure_futures_shadow(data_dir, now=now, cfg=cfg)
    symbol = str(spot_open.get("symbol") or "")
    slot_id = int(spot_open.get("slot_id") or 0)
    reason = None
    if not spot_open.get("opened") or not symbol or not slot_id:
        reason = "SPOT_ENTRY_NOT_ACCEPTED"
    elif not snapshot:
        reason = "NO_USDM_PERPETUAL_QUOTE"
    slot = next((item for item in state["slots"] if item["id"] == slot_id), None)
    if reason is None and slot is None:
        reason = "INVALID_SLOT"
    elif reason is None and slot.get("position"):
        reason = "SLOT_ALREADY_OPEN"

    ask = _finite_positive((snapshot or {}).get("ask"))
    bid = _finite_positive((snapshot or {}).get("bid"))
    mark = _finite_positive((snapshot or {}).get("mark"))
    if reason is None and (not ask or not bid or not mark):
        reason = "INVALID_USDM_QUOTE"
    if reason is not None:
        skipped = {
            "observed_at": _timestamp(now),
            "symbol": symbol,
            "slot_id": slot_id or None,
            "reason": reason,
        }
        _append_bounded(state.setdefault("skipped_signals", []), skipped)
        state["updated_at"] = _timestamp(now)
        _write_json(data_dir / STATE_FILENAME, state)
        return {"opened": False, **skipped}

    allocation = float(slot["balance"])
    notional = allocation * cfg.leverage
    quantity = notional / ask
    entry_fee = notional * cfg.taker_fee_rate
    spot_price = _finite_positive(spot_open.get("entry_price"))
    position = {
        "alert_id": spot_open.get("alert_id"),
        "symbol": symbol,
        "slot_id": slot_id,
        "opened_at": _timestamp(now),
        "entry_price": ask,
        "entry_bid": bid,
        "entry_mark": mark,
        "spot_entry_price": spot_price or None,
        "entry_basis": ask / spot_price - 1.0 if spot_price else None,
        "entry_spread": ask / bid - 1.0,
        "entry_allocation": allocation,
        "notional": notional,
        "quantity": quantity,
        "entry_fee": entry_fee,
        "funding_pnl": 0.0,
        "current_bid": bid,
        "current_mark": mark,
        "last_funding_rate": float(snapshot.get("last_funding_rate") or 0.0),
        "next_funding_time": snapshot.get("next_funding_time"),
        "updated_at": _timestamp(now),
    }
    slot["balance"] = 0.0
    slot["position"] = position
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return {"opened": True, **position}


def _record_sample(data_dir: Path, position: dict, snapshot: dict, now: datetime) -> None:
    index = _finite_positive(snapshot.get("index"))
    mark = _finite_positive(snapshot.get("mark"))
    sample = {
        "observed_at": _timestamp(now),
        "slot_id": position["slot_id"],
        "symbol": position["symbol"],
        "bid": snapshot.get("bid"),
        "ask": snapshot.get("ask"),
        "mark": mark or None,
        "index": index or None,
        "basis": mark / index - 1.0 if mark and index else None,
        "last_funding_rate": snapshot.get("last_funding_rate"),
        "next_funding_time": snapshot.get("next_funding_time"),
    }
    path = data_dir / SAMPLES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, separators=(",", ":")) + "\n")


def _apply_funding(
    state: dict,
    position: dict,
    snapshot: dict,
    *,
    now: datetime,
) -> None:
    previous_time = int(position.get("next_funding_time") or 0)
    next_time = int(snapshot.get("next_funding_time") or 0)
    now_ms = int(now.timestamp() * 1000)
    if previous_time and now_ms >= previous_time and next_time > previous_time:
        rate = float(snapshot.get("last_funding_rate") or 0.0)
        mark = _finite_positive(snapshot.get("mark"))
        funding_pnl = -float(position["quantity"]) * mark * rate
        position["funding_pnl"] = float(position.get("funding_pnl") or 0.0) + funding_pnl
        _append_bounded(
            state.setdefault("funding_events", []),
            {
                "observed_at": _timestamp(now),
                "funding_time": previous_time,
                "slot_id": position["slot_id"],
                "symbol": position["symbol"],
                "rate": rate,
                "mark": mark,
                "pnl": funding_pnl,
            },
        )
    if next_time:
        position["next_funding_time"] = next_time


def update_futures_shadow(
    data_dir: Path,
    snapshots: dict[str, dict],
    spot_closed: list[dict],
    *,
    now: datetime,
    cfg: FuturesShadowConfig = DEFAULT_FUTURES_CONFIG,
) -> list[dict]:
    """Mark shadow positions and mirror spot exits at the perpetual bid."""
    state = ensure_futures_shadow(data_dir, now=now, cfg=cfg)
    closed_by_slot = {int(item["slot_id"]): item for item in spot_closed}
    closed: list[dict] = []
    for slot in state["slots"]:
        position = slot.get("position")
        if not position:
            continue
        snapshot = snapshots.get(str(position["symbol"]))
        if snapshot:
            _apply_funding(state, position, snapshot, now=now)
            position["current_bid"] = _finite_positive(snapshot.get("bid"))
            position["current_mark"] = _finite_positive(snapshot.get("mark"))
            position["last_funding_rate"] = float(
                snapshot.get("last_funding_rate") or 0.0
            )
            position["updated_at"] = _timestamp(now)
            _record_sample(data_dir, position, snapshot, now)

        spot_trade = closed_by_slot.get(int(slot["id"]))
        if not spot_trade:
            continue
        exit_price = _finite_positive((snapshot or {}).get("bid"))
        quote_quality = "LIVE_BID"
        if not exit_price:
            exit_price = _finite_positive(position.get("current_bid"))
            quote_quality = "LAST_SEEN_BID"
        if not exit_price:
            continue
        price_pnl = float(position["quantity"]) * (
            exit_price - float(position["entry_price"])
        )
        exit_notional = float(position["quantity"]) * exit_price
        exit_fee = exit_notional * cfg.taker_fee_rate
        funding_pnl = float(position.get("funding_pnl") or 0.0)
        allocation = float(position["entry_allocation"])
        pnl = price_pnl - float(position["entry_fee"]) - exit_fee + funding_pnl
        trade = {
            **position,
            "closed_at": _timestamp(now),
            "exit_price": exit_price,
            "exit_fee": exit_fee,
            "exit_reason": spot_trade.get("exit_reason"),
            "spot_exit_price": spot_trade.get("exit_price"),
            "quote_quality": quote_quality,
            "price_pnl": price_pnl,
            "pnl": pnl,
            "return": pnl / allocation if allocation else 0.0,
            "ending_balance": allocation + pnl,
        }
        slot["balance"] = allocation + pnl
        slot["position"] = None
        _append_bounded(state.setdefault("closed_trades", []), trade)
        closed.append(trade)
    state["updated_at"] = _timestamp(now)
    _write_json(data_dir / STATE_FILENAME, state)
    return closed


def futures_shadow_payload(
    data_dir: Path,
    *,
    cfg: FuturesShadowConfig = DEFAULT_FUTURES_CONFIG,
) -> dict:
    state = ensure_futures_shadow(data_dir, cfg=cfg)
    cash = 0.0
    market_value = 0.0
    open_positions = []
    for slot in state["slots"]:
        position = slot.get("position")
        if not position:
            cash += float(slot.get("balance") or 0.0)
            continue
        bid = _finite_positive(position.get("current_bid"))
        price_pnl = float(position["quantity"]) * (
            bid - float(position["entry_price"])
        )
        estimated_exit_fee = (
            float(position["quantity"]) * bid * cfg.taker_fee_rate
        )
        pnl = (
            price_pnl
            - float(position["entry_fee"])
            - estimated_exit_fee
            + float(position.get("funding_pnl") or 0.0)
        )
        value = float(position["entry_allocation"]) + pnl
        market_value += value
        open_positions.append(
            {
                **position,
                "estimated_exit_fee": estimated_exit_fee,
                "unrealized_pnl": pnl,
                "current_return": pnl / float(position["entry_allocation"]),
                "margin_value": value,
            }
        )
    equity = cash + market_value
    closed = list(reversed(state.get("closed_trades", [])))
    realized_pnl = sum(float(item.get("pnl") or 0.0) for item in closed)
    total_fees = sum(
        float(item.get("entry_fee") or 0.0) + float(item.get("exit_fee") or 0.0)
        for item in closed
    ) + sum(
        float(item.get("entry_fee") or 0.0)
        + float(item.get("estimated_exit_fee") or 0.0)
        for item in open_positions
    )
    funding_pnl = sum(float(item.get("pnl") or 0.0) for item in state.get("funding_events", []))
    return {
        "mode": "PAPER_ONLY_NO_ORDERS",
        "started_at": state.get("created_at"),
        "starting_equity": cfg.starting_equity,
        "equity": equity,
        "total_return": equity / cfg.starting_equity - 1.0,
        "cash": cash,
        "market_value": market_value,
        "realized_pnl": realized_pnl,
        "fees": total_fees,
        "funding_pnl": funding_pnl,
        "open_count": len(open_positions),
        "available_slots": cfg.max_positions - len(open_positions),
        "max_positions": cfg.max_positions,
        "config": asdict(cfg),
        "open_positions": open_positions,
        "closed_trades": closed[:100],
        "skipped_signals": list(reversed(state.get("skipped_signals", [])))[:100],
        "funding_events": list(reversed(state.get("funding_events", [])))[:100],
        "updated_at": state.get("updated_at"),
    }
