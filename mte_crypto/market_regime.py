from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import median

import pandas as pd


UTC = timezone.utc


@dataclass(frozen=True)
class BullRegimeConfig:
    min_liquid_quote_volume: float = 1_000_000.0
    min_positive_breadth: float = 0.65
    min_median_return_24h: float = 0.02
    min_breakout_day_return: float = 0.03
    breakout_lookback_days: int = 20
    hold_buffer: float = 0.02


DEFAULT_BULL_REGIME_CONFIG = BullRegimeConfig()


def detect_bull_regime(
    markets: list[dict],
    btc_daily: pd.DataFrame,
    *,
    now: datetime,
    cfg: BullRegimeConfig = DEFAULT_BULL_REGIME_CONFIG,
) -> dict:
    """Detect a broad, causal BTC breakout without waiting for a narrative label."""
    liquid_returns = [
        float(row.get("return_24h") or 0.0)
        for row in markets
        if float(row.get("quote_volume_24h") or 0.0)
        >= cfg.min_liquid_quote_volume
    ]
    breadth = (
        sum(value > 0.0 for value in liquid_returns) / len(liquid_returns)
        if liquid_returns
        else 0.0
    )
    median_return = median(liquid_returns) if liquid_returns else 0.0

    result = {
        "active": False,
        "state": "NORMAL",
        "observed_at": now.astimezone(UTC).isoformat(),
        "liquid_pairs": len(liquid_returns),
        "positive_breadth": breadth,
        "median_return_24h": median_return,
        "btc_price": None,
        "btc_breakout_level": None,
        "btc_session_return": None,
        "current_breakout": False,
        "confirmed_breakout": False,
        "config": asdict(cfg),
    }
    if btc_daily.empty or len(btc_daily) < cfg.breakout_lookback_days + 1:
        return result

    frame = btc_daily.sort_index().copy()
    now_ts = pd.Timestamp(now.astimezone(UTC))
    closed = frame[frame["close_time"] <= now_ts]
    current = frame.iloc[-1]
    btc_price = float(current["close"])
    btc_session_return = (
        btc_price / float(current["open"]) - 1.0
        if float(current["open"]) > 0
        else 0.0
    )

    prior_closed = closed.tail(cfg.breakout_lookback_days)
    if len(prior_closed) < cfg.breakout_lookback_days:
        return result
    breakout_level = float(prior_closed["high"].max())
    current_is_open = pd.Timestamp(current["close_time"]) > now_ts
    current_breakout = (
        current_is_open
        and btc_price > breakout_level
        and btc_session_return >= cfg.min_breakout_day_return
    )

    confirmed_breakout = False
    if len(closed) >= cfg.breakout_lookback_days + 1:
        latest_closed = closed.iloc[-1]
        prior = closed.iloc[-(cfg.breakout_lookback_days + 1) : -1]
        confirmed_breakout = float(latest_closed["close"]) > float(prior["high"].max())
        if confirmed_breakout:
            breakout_level = float(prior["high"].max())

    broad_risk_on = (
        breadth >= cfg.min_positive_breadth
        and median_return >= cfg.min_median_return_24h
    )
    holds_breakout = btc_price >= breakout_level * (1.0 - cfg.hold_buffer)
    active = broad_risk_on and holds_breakout and (
        current_breakout or confirmed_breakout
    )
    result.update(
        {
            "active": active,
            "state": "BULL_BREAKOUT" if active else "NORMAL",
            "btc_price": btc_price,
            "btc_breakout_level": breakout_level,
            "btc_session_return": btc_session_return,
            "current_breakout": current_breakout,
            "confirmed_breakout": confirmed_breakout,
        }
    )
    return result
