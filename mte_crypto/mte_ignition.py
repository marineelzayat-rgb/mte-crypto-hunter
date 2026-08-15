from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import true_range, wilder_atr


@dataclass
class LineCandidate:
    resistance: bool
    y1: float
    t1: float
    y2: float
    t2: float
    level: int
    base_score: float
    created_bar: int
    live_touches: int = 0
    last_touch_bar: int = -1_000_000
    broken: bool = False
    selected_prev: bool = False

    def project(self, timestamp: float) -> float:
        if self.t2 == self.t1:
            return np.nan
        return self.y2 + (self.y2 - self.y1) / (self.t2 - self.t1) * (timestamp - self.t2)


@dataclass(frozen=True)
class MTEConfig:
    atr_len: int = 14
    break_buffer_atr: float = 0.10
    pivot_line_tolerance_atr: float = 0.18
    touch_tolerance_atr: float = 0.22
    live_touch_tolerance_atr: float = 0.18
    live_touch_cooldown: int = 3
    min_bars_after_creation: int = 1
    max_candidates_per_side: int = 160
    ladder_lines_per_side: int = 4
    cluster_tolerance_atr: float = 0.28
    h4_pivot_strength: int = 4
    h4_pivots_kept: int = 12
    h4_min_span: int = 5
    h4_min_move_atr: float = 0.55
    d1_pivot_strength: int = 3
    d1_pivots_kept: int = 14
    d1_min_span: int = 4
    d1_min_move_atr: float = 0.65
    h1_direction_pivot_strength: int = 3
    h1_direction_buffer_atr: float = 0.10
    min_tr_atr: float = 1.05
    min_body_ratio: float = 0.55
    min_clv: float = 0.68
    min_di_spread: float = 4.0


DEFAULT_MTE_CONFIG = MTEConfig()


def _time_number(index: pd.DatetimeIndex) -> np.ndarray:
    return index.asi8.astype(float) / 3_600_000_000_000.0


def _aggregate_complete(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    keys = frame.index.floor(f"{hours}h")
    grouped = frame.groupby(keys)
    out = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
    )
    counts = grouped["close"].count()
    last_times = grouped.apply(lambda x: x.index[-1], include_groups=False)
    out = out[counts >= hours].copy()
    out.index = pd.DatetimeIndex(last_times.loc[out.index])
    return out.sort_index()


def _pivot_events(source: pd.DataFrame, strength: int) -> dict[pd.Timestamp, list[dict]]:
    events: dict[pd.Timestamp, list[dict]] = {}
    atr = wilder_atr(source, 14)
    times = _time_number(source.index)
    highs = source["high"].to_numpy(float)
    lows = source["low"].to_numpy(float)
    for confirm in range(2 * strength, len(source)):
        pivot = confirm - strength
        lo = pivot - strength
        hi = pivot + strength + 1
        high_window = highs[lo:hi]
        low_window = lows[lo:hi]
        timestamp = source.index[confirm]
        bucket = events.setdefault(timestamp, [])
        if np.isfinite(high_window).all() and highs[pivot] >= high_window.max():
            bucket.append({"kind": "high", "price": highs[pivot], "time": times[pivot], "atr": float(atr.iloc[pivot]), "source_i": pivot})
        if np.isfinite(low_window).all() and lows[pivot] <= low_window.min():
            bucket.append({"kind": "low", "price": lows[pivot], "time": times[pivot], "atr": float(atr.iloc[pivot]), "source_i": pivot})
    return events


def _dmi(frame: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series]:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = true_range(frame).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    plus = 100.0 * plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr
    minus = 100.0 * minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr
    return plus, minus


def compute_mte_ignitions(frame: pd.DataFrame, cfg: MTEConfig = DEFAULT_MTE_CONFIG) -> pd.DataFrame:
    """Causal long-side signal port of MTE v3.0.2's H4/D1 selected-line engine."""
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    h4 = _aggregate_complete(frame, 4)
    d1 = _aggregate_complete(frame, 24)
    event_maps = {
        2: _pivot_events(h4, cfg.h4_pivot_strength),
        3: _pivot_events(d1, cfg.d1_pivot_strength),
    }
    h1_events = _pivot_events(frame, cfg.h1_direction_pivot_strength)

    atr = wilder_atr(frame, cfg.atr_len)
    plus_di, minus_di = _dmi(frame, cfg.atr_len)
    tr = true_range(frame)
    candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    tr_atr = tr / atr.shift(1)
    body = (frame["close"] - frame["open"]).abs() / candle_range
    bull_clv = (frame["close"] - frame["low"]) / candle_range
    bear_clv = (frame["high"] - frame["close"]) / candle_range
    time_values = _time_number(frame.index)

    banks: dict[tuple[int, str], list[dict]] = {(2, "high"): [], (2, "low"): [], (3, "high"): [], (3, "low"): []}
    candidates: dict[bool, list[LineCandidate]] = {True: [], False: []}
    exact_keys: set[tuple] = set()
    last_h1_high = np.nan
    last_h1_low = np.nan
    trend_lock = 0
    rows: list[dict] = []

    def params(level: int) -> tuple[int, int, float, float]:
        if level == 2:
            return cfg.h4_pivots_kept, cfg.h4_min_span, cfg.h4_min_move_atr, 2.0
        return cfg.d1_pivots_kept, cfg.d1_min_span, cfg.d1_min_move_atr, 4.0

    def trim(side: bool) -> None:
        pool = candidates[side]
        while len(pool) > cfg.max_candidates_per_side:
            broken = [i for i, candidate in enumerate(pool) if candidate.broken]
            idx = broken[0] if broken else min(range(len(pool)), key=lambda i: pool[i].base_score)
            exact_keys.discard((pool[idx].resistance, pool[idx].level, pool[idx].t1, pool[idx].t2, pool[idx].y1, pool[idx].y2))
            pool.pop(idx)

    def add_pivot(level: int, pivot: dict, chart_i: int, close: float, safe_atr: float) -> None:
        kind = pivot["kind"]
        bank = banks[(level, kind)]
        kept, min_span, min_move, tf_base = params(level)
        for old_pos, old in enumerate(bank):
            descending = kind == "high" and pivot["price"] < old["price"]
            ascending = kind == "low" and pivot["price"] > old["price"]
            span = pivot["source_i"] - old["source_i"]
            if not (descending or ascending) or span < min_span:
                continue
            atr_values = [value for value in (old["atr"], pivot["atr"]) if np.isfinite(value)]
            if not atr_values:
                continue
            ref_atr = float(np.mean(atr_values))
            if ref_atr <= 0:
                continue
            move_atr = abs(pivot["price"] - old["price"]) / ref_atr
            if move_atr < min_move:
                continue
            valid = True
            touches = 0
            for mid in bank[old_pos + 1 :]:
                slope = (pivot["price"] - old["price"]) / (pivot["time"] - old["time"])
                line_mid = pivot["price"] + slope * (mid["time"] - pivot["time"])
                tol = cfg.pivot_line_tolerance_atr * max(mid["atr"], 1e-12)
                touch_tol = cfg.touch_tolerance_atr * max(mid["atr"], 1e-12)
                if (kind == "high" and mid["price"] > line_mid + tol) or (kind == "low" and mid["price"] < line_mid - tol):
                    valid = False
                    break
                touches += int(abs(mid["price"] - line_mid) <= touch_tol)
            if not valid:
                continue
            key = (kind == "high", level, old["time"], pivot["time"], old["price"], pivot["price"])
            if key in exact_keys:
                continue
            base = tf_base + min(move_atr, 5.0) + 0.70 * min(span / max(min_span, 1), 5.0) + 1.10 * touches
            candidate = LineCandidate(kind == "high", old["price"], old["time"], pivot["price"], pivot["time"], level, base, chart_i)
            projected = candidate.project(time_values[chart_i])
            valid_side = close <= projected + cfg.break_buffer_atr * safe_atr if candidate.resistance else close >= projected - cfg.break_buffer_atr * safe_atr
            if np.isfinite(projected) and valid_side:
                candidates[candidate.resistance].append(candidate)
                exact_keys.add(key)
                trim(candidate.resistance)
        bank.append(pivot)
        del bank[:-kept]

    def select(side: bool, chart_i: int, close: float, safe_atr: float) -> list[LineCandidate]:
        chosen: list[LineCandidate] = []
        now = time_values[chart_i]
        for _ in range(cfg.ladder_lines_per_side):
            best = None
            best_rank = -1e18
            for candidate in candidates[side]:
                if candidate.broken or candidate in chosen:
                    continue
                y = candidate.project(now)
                eligible = np.isfinite(y) and (y > close if side else y < close)
                if not eligible:
                    continue
                if any(abs(existing.project(now) - y) <= cfg.cluster_tolerance_atr * safe_atr for existing in chosen):
                    continue
                distance = (y - close) / safe_atr if side else (close - y) / safe_atr
                age_days = max((now - candidate.t2) / 24.0, 0.0)
                rank = candidate.base_score + 0.80 * candidate.live_touches + 0.025 * min(age_days, 120.0) - 0.45 * max(distance, 0.0)
                if rank > best_rank:
                    best_rank = rank
                    best = candidate
            if best is not None:
                chosen.append(best)
        return chosen

    for i, timestamp in enumerate(frame.index):
        close = float(frame["close"].iloc[i])
        safe_atr = max(float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else 0.0, 1e-12)
        prev_atr = max(float(atr.iloc[i - 1]) if i and np.isfinite(atr.iloc[i - 1]) else safe_atr, 1e-12)

        for event in h1_events.get(timestamp, []):
            if event["kind"] == "high":
                last_h1_high = event["price"]
            else:
                last_h1_low = event["price"]
        for level in (2, 3):
            for event in event_maps[level].get(timestamp, []):
                add_pivot(level, event, i, close, safe_atr)

        up_levels: list[int] = []
        down_levels: list[int] = []
        broken_up_prices: list[float] = []
        for side in (True, False):
            for candidate in candidates[side]:
                if candidate.broken or i == 0:
                    continue
                y_now = candidate.project(time_values[i])
                y_prev = candidate.project(time_values[i - 1])
                age_ok = i - candidate.created_bar >= cfg.min_bars_after_creation
                crossed = age_ok and np.isfinite(y_now) and np.isfinite(y_prev)
                if side:
                    crossed = crossed and close > y_now + cfg.break_buffer_atr * safe_atr and float(frame["close"].iloc[i - 1]) <= y_prev + cfg.break_buffer_atr * prev_atr
                else:
                    crossed = crossed and close < y_now - cfg.break_buffer_atr * safe_atr and float(frame["close"].iloc[i - 1]) >= y_prev - cfg.break_buffer_atr * prev_atr
                if crossed:
                    was_selected = candidate.selected_prev
                    candidate.broken = True
                    if was_selected:
                        (up_levels if side else down_levels).append(candidate.level)
                        if side:
                            broken_up_prices.append(y_now)

        for side in (True, False):
            for candidate in candidates[side]:
                if candidate.broken:
                    continue
                y = candidate.project(time_values[i])
                if side:
                    touch = np.isfinite(y) and float(frame["high"].iloc[i]) >= y - cfg.live_touch_tolerance_atr * safe_atr and close <= y + cfg.break_buffer_atr * safe_atr
                else:
                    touch = np.isfinite(y) and float(frame["low"].iloc[i]) <= y + cfg.live_touch_tolerance_atr * safe_atr and close >= y - cfg.break_buffer_atr * safe_atr
                if touch and i - candidate.last_touch_bar >= cfg.live_touch_cooldown:
                    candidate.live_touches += 1
                    candidate.last_touch_bar = i

        bull_quality = bool(
            tr_atr.iloc[i] >= cfg.min_tr_atr and body.iloc[i] >= cfg.min_body_ratio
            and bull_clv.iloc[i] >= cfg.min_clv and plus_di.iloc[i] - minus_di.iloc[i] >= cfg.min_di_spread
        )
        bear_quality = bool(
            tr_atr.iloc[i] >= cfg.min_tr_atr and body.iloc[i] >= cfg.min_body_ratio
            and bear_clv.iloc[i] >= cfg.min_clv and minus_di.iloc[i] - plus_di.iloc[i] >= cfg.min_di_spread
        )
        bull_raw = bool(up_levels and bull_quality)
        bear_raw = bool(down_levels and bear_quality)
        h1_buffer = cfg.h1_direction_buffer_atr * safe_atr
        prev_close = float(frame["close"].iloc[i - 1]) if i else close
        bear_flip = np.isfinite(last_h1_low) and close < last_h1_low - h1_buffer and prev_close >= last_h1_low - h1_buffer
        bull_flip = np.isfinite(last_h1_high) and close > last_h1_high + h1_buffer and prev_close <= last_h1_high + h1_buffer
        if trend_lock == 0:
            if bull_raw and not bear_raw:
                trend_lock = 1
            elif bear_raw and not bull_raw:
                trend_lock = -1
        elif trend_lock == 1 and bear_flip:
            trend_lock = -1
        elif trend_lock == -1 and bull_flip:
            trend_lock = 1
        bull_signal = bull_raw and trend_lock == 1
        line_price = max(broken_up_prices) if broken_up_prices else np.nan
        stop_price = line_price - 0.30 * safe_atr if np.isfinite(line_price) else np.nan
        risk_pct = (close - stop_price) / close if np.isfinite(stop_price) and close > 0 else np.nan

        selected_r = select(True, i, close, safe_atr)
        selected_s = select(False, i, close, safe_atr)
        for candidate in candidates[True]:
            candidate.selected_prev = candidate in selected_r and not candidate.broken
        for candidate in candidates[False]:
            candidate.selected_prev = candidate in selected_s and not candidate.broken

        rows.append(
            {
                "bull_ignition": bull_signal,
                "bull_raw": bull_raw,
                "bull_quality": bull_quality,
                "trend_lock": trend_lock,
                "broken_h4": int(2 in up_levels),
                "broken_d1": int(3 in up_levels),
                "broken_line": line_price,
                "structural_stop": stop_price,
                "risk_pct": risk_pct,
                "last_h1_pivot_low": last_h1_low,
                "atr": safe_atr,
            }
        )
    return pd.DataFrame(rows, index=frame.index)
