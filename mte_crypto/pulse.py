from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median


UTC = timezone.utc


@dataclass(frozen=True)
class PulseConfig:
    history_minutes: int = 90
    min_return_5m: float = 0.015
    min_return_15m: float = 0.025
    strong_return_5m: float = 0.04
    min_quote_delta_5m: float = 75_000.0
    min_volume_acceleration: float = 3.0
    max_early_return_24h: float = 0.20
    max_bull_continuation_return_24h: float = 0.40
    max_bull_continuation_return_5m: float = 0.08
    max_bull_continuation_return_15m: float = 0.18
    max_research_return_24h: float = 0.45
    ttl_minutes: int = 120


DEFAULT_PULSE_CONFIG = PulseConfig()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _sample_at_or_before(samples: list[dict], cutoff: datetime) -> dict | None:
    eligible = [sample for sample in samples if _parse_time(sample["at"]) <= cutoff]
    return eligible[-1] if eligible else None


def _return(current: float, prior: dict | None) -> float | None:
    if not prior:
        return None
    old = float(prior.get("price") or 0.0)
    return current / old - 1.0 if old > 0 else None


def _prior_volume_velocity(samples: list[dict]) -> float:
    deltas: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        elapsed = (_parse_time(current["at"]) - _parse_time(previous["at"])).total_seconds()
        if elapsed <= 0:
            continue
        change = float(current["quote_volume_24h"]) - float(previous["quote_volume_24h"])
        if change > 0:
            deltas.append(change / elapsed)
    return median(deltas) if deltas else 0.0


def evaluate_pulses(
    current_rows: list[dict],
    history: dict[str, list[dict]],
    *,
    now: datetime,
    cfg: PulseConfig = DEFAULT_PULSE_CONFIG,
    bull_mode: bool = False,
) -> list[dict]:
    """Find fast price/volume wakeups without pre-filtering by 24h liquidity."""
    candidates: list[dict] = []
    for market in current_rows:
        symbol = str(market.get("symbol", ""))
        price = float(market.get("last_price") or 0.0)
        return_24h = float(market.get("return_24h") or 0.0)
        quote_volume = float(market.get("quote_volume_24h") or 0.0)
        samples = history.get(symbol, [])
        if not symbol or price <= 0 or not samples:
            continue

        prior_5m = _sample_at_or_before(samples, now - timedelta(minutes=4))
        prior_15m = _sample_at_or_before(samples, now - timedelta(minutes=14))
        return_5m = _return(price, prior_5m)
        return_15m = _return(price, prior_15m)
        if return_5m is None or prior_5m is None:
            continue

        quote_delta_5m = max(
            0.0,
            quote_volume - float(prior_5m.get("quote_volume_24h") or 0.0),
        )
        elapsed = max((now - _parse_time(prior_5m["at"])).total_seconds(), 1.0)
        current_velocity = quote_delta_5m / elapsed
        baseline_velocity = _prior_volume_velocity(samples)
        volume_acceleration = (
            current_velocity / baseline_velocity if baseline_velocity > 0 else None
        )

        fast_price = return_5m >= cfg.min_return_5m or (
            return_15m is not None and return_15m >= cfg.min_return_15m
        )
        volume_surge = quote_delta_5m >= cfg.min_quote_delta_5m and (
            (
                volume_acceleration is not None
                and volume_acceleration >= cfg.min_volume_acceleration
            )
            or return_5m >= cfg.strong_return_5m
        )
        if not fast_price or not volume_surge or return_24h > cfg.max_research_return_24h:
            continue

        if return_24h <= cfg.max_early_return_24h:
            state = "EARLY_PULSE"
        elif (
            bull_mode
            and return_24h <= cfg.max_bull_continuation_return_24h
            and return_5m <= cfg.max_bull_continuation_return_5m
            and (
                return_15m is None
                or return_15m <= cfg.max_bull_continuation_return_15m
            )
        ):
            state = "BULL_CONTINUATION"
        else:
            state = "RAPID_MOVE_NO_CHASE"
        candidates.append(
            {
                "symbol": symbol,
                "state": state,
                "price": price,
                "return_24h": return_24h,
                "quote_volume_24h": quote_volume,
                "return_5m": return_5m,
                "return_15m": return_15m,
                "quote_volume_delta_5m": quote_delta_5m,
                "volume_acceleration": volume_acceleration,
                "bull_mode": bull_mode,
            }
        )
    priority = {
        "EARLY_PULSE": 0,
        "BULL_CONTINUATION": 0 if bull_mode else 1,
        "RAPID_MOVE_NO_CHASE": 2,
    }
    return sorted(
        candidates,
        key=lambda row: (priority.get(row["state"], 9), -row["return_5m"]),
    )


def append_history(
    current_rows: list[dict],
    history: dict[str, list[dict]],
    *,
    now: datetime,
    history_minutes: int,
) -> dict[str, list[dict]]:
    cutoff = now - timedelta(minutes=history_minutes)
    next_history: dict[str, list[dict]] = {}
    for market in current_rows:
        symbol = str(market.get("symbol", ""))
        price = float(market.get("last_price") or 0.0)
        if not symbol or price <= 0:
            continue
        samples = [
            sample
            for sample in history.get(symbol, [])
            if _parse_time(sample["at"]) >= cutoff
        ]
        samples.append(
            {
                "at": now.isoformat(),
                "price": price,
                "quote_volume_24h": float(market.get("quote_volume_24h") or 0.0),
            }
        )
        next_history[symbol] = samples
    return next_history


def update_pulse_candidates(
    rows: list[dict],
    active: dict[str, dict],
    *,
    now: datetime,
    ttl_minutes: int,
) -> tuple[dict[str, dict], list[dict]]:
    active = {
        symbol: item
        for symbol, item in active.items()
        if _parse_time(item["expires_at"]) > now
    }
    new_alerts: list[dict] = []
    for row in rows:
        symbol = row["symbol"]
        if symbol not in active:
            new_alerts.append(row)
        active[symbol] = {
            "detected_at": active.get(symbol, {}).get("detected_at", now.isoformat()),
            "expires_at": (now + timedelta(minutes=ttl_minutes)).isoformat(),
            "state": row["state"],
            "price": row["price"],
        }
    return active, new_alerts


def format_pulse_alert(row: dict) -> str:
    acceleration = row.get("volume_acceleration", 0.0)
    acceleration_text = "new baseline" if acceleration is None else f"{acceleration:.1f}x"
    return_15m = row.get("return_15m")
    return_15m_text = f"{100 * return_15m:.1f}%" if return_15m is not None else "warming up"
    if row["state"] == "EARLY_PULSE":
        action = "Early wakeup — Wave Rider paper entry"
    elif row["state"] == "BULL_CONTINUATION":
        action = "Bull-regime continuation — controlled Wave Rider paper entry"
    else:
        action = "Rapid move too extended — order-book research only, NO CHASE"
    return (
        f"MTE CRYPTO — {row['state']}\n"
        f"{row['symbol']} @ {row['price']:.8g}\n"
        f"5m return: {100 * row['return_5m']:.1f}%\n"
        f"15m return: {return_15m_text}\n"
        f"24h return: {100 * row['return_24h']:.1f}%\n"
        f"5m new quote volume: ${row['quote_volume_delta_5m']:,.0f}\n"
        f"Volume acceleration: {acceleration_text}\n"
        f"24h quote volume: ${row['quote_volume_24h']:,.0f}\n"
        f"Action: {action}"
    )
