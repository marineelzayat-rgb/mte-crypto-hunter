from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


UTC = timezone.utc
CHECKPOINT_HOURS = (6, 12, 24, 48)
PUBLIC_FIELDS = (
    "symbol",
    "state",
    "price",
    "return_24h",
    "quote_volume_24h",
    "rvol_1h",
    "volume_24h_ratio",
    "relative_strength_24h",
    "structural_risk_pct",
    "break_distance_atr",
    "hunter_probability",
    "return_5m",
    "return_15m",
    "quote_volume_delta_5m",
    "volume_acceleration",
)
ACTIVE_FIELDS = ("detected_at", "expires_at", "state", "price", "hunter_probability")


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


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def public_alert(row: dict, *, source: str, observed_at: datetime) -> dict:
    timestamp = observed_at.astimezone(UTC).isoformat()
    record = {
        "id": f"{timestamp}|{row.get('symbol', '')}|{row.get('state', '')}",
        "observed_at": timestamp,
        "source": source,
    }
    for field in PUBLIC_FIELDS:
        if field in row:
            record[field] = _clean(row[field])
    return record


def record_alert(
    data_dir: Path,
    row: dict,
    *,
    source: str,
    observed_at: datetime,
) -> dict:
    """Append one public, secret-free alert and initialize its outcome record."""
    data_dir.mkdir(parents=True, exist_ok=True)
    record = public_alert(row, source=source, observed_at=observed_at)
    with (data_dir / "alerts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    outcomes_path = data_dir / "alert_outcomes.json"
    outcomes = _read_json(outcomes_path)
    price = float(record.get("price") or 0.0)
    outcomes.setdefault(
        record["id"],
        {
            "symbol": record.get("symbol"),
            "state": record.get("state"),
            "source": source,
            "observed_at": record["observed_at"],
            "start_price": price,
            "current_price": price,
            "current_return": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
            "checkpoints": {},
            "updated_at": record["observed_at"],
        },
    )
    _write_json(outcomes_path, outcomes)
    return record


def bootstrap_active_alerts(data_dir: Path) -> None:
    """Seed a new ledger from the already-persisted active candidate files."""
    ledger = data_dir / "alerts.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        return
    sources = (
        ("active_candidates.json", "hunter", "HUNTER_ALERT"),
        ("pulse_candidates.json", "pulse", None),
    )
    for filename, source, default_state in sources:
        for symbol, item in _read_json(data_dir / filename).items():
            detected_at = item.get("detected_at")
            try:
                observed_at = datetime.fromisoformat(str(detected_at))
            except (TypeError, ValueError):
                observed_at = datetime.now(UTC)
            if not observed_at.tzinfo:
                observed_at = observed_at.replace(tzinfo=UTC)
            row = {
                "symbol": symbol,
                "state": item.get("state") or default_state,
                "price": item.get("price"),
                "hunter_probability": item.get("hunter_probability"),
            }
            record_alert(data_dir, row, source=source, observed_at=observed_at)


def update_alert_outcomes(
    data_dir: Path,
    markets: list[dict],
    *,
    now: datetime,
) -> None:
    outcomes_path = data_dir / "alert_outcomes.json"
    outcomes = _read_json(outcomes_path)
    if not outcomes:
        return
    prices = {
        str(row.get("symbol", "")): float(row.get("last_price") or 0.0)
        for row in markets
    }
    changed = False
    now_utc = now.astimezone(UTC)
    for item in outcomes.values():
        symbol = str(item.get("symbol", ""))
        price = prices.get(symbol, 0.0)
        start = float(item.get("start_price") or 0.0)
        if price <= 0 or start <= 0:
            continue
        observed = datetime.fromisoformat(item["observed_at"])
        if not observed.tzinfo:
            observed = observed.replace(tzinfo=UTC)
        elapsed_hours = (now_utc - observed).total_seconds() / 3600.0
        current_return = price / start - 1.0
        item["current_price"] = price
        item["current_return"] = current_return
        item["max_return"] = max(float(item.get("max_return", 0.0)), current_return)
        item["min_return"] = min(float(item.get("min_return", 0.0)), current_return)
        item["updated_at"] = now_utc.isoformat()
        checkpoints = item.setdefault("checkpoints", {})
        for hours in CHECKPOINT_HOURS:
            key = f"{hours}h"
            if elapsed_hours >= hours and key not in checkpoints:
                checkpoints[key] = {
                    "price": price,
                    "return": current_return,
                    "captured_at": now_utc.isoformat(),
                }
        changed = True
    if changed:
        _write_json(outcomes_path, outcomes)


def read_alerts(path: Path, *, limit: int = 100) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records[-limit:]


def status_payload(data_dir: Path, *, limit: int = 100) -> dict:
    alerts = read_alerts(data_dir / "alerts.jsonl", limit=limit)
    outcomes = _read_json(data_dir / "alert_outcomes.json")
    merged = []
    for alert in reversed(alerts):
        item = dict(alert)
        item["outcome"] = outcomes.get(alert.get("id"), {})
        merged.append(item)
    def public_active(filename: str) -> dict:
        result = {}
        for symbol, item in _read_json(data_dir / filename).items():
            result[symbol] = {
                field: _clean(item[field]) for field in ACTIVE_FIELDS if field in item
            }
        return result

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "disclaimer": "Discovery research only. No order placement.",
        "active": {
            "hunter": public_active("active_candidates.json"),
            "pulse": public_active("pulse_candidates.json"),
        },
        "alerts": merged,
    }
