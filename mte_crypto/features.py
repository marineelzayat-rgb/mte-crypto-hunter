from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import ScannerConfig


def _safe_div(a, b):
    return a / b.replace(0.0, np.nan) if isinstance(b, pd.Series) else a / (b if b else np.nan)


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def wilder_atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(frame).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def percentile_rank_last(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling percentile rank of the current observation."""
    def rank_last(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if clean.size < 8:
            return np.nan
        return float(np.mean(clean <= clean[-1]))

    return series.rolling(window, min_periods=max(8, window // 3)).apply(rank_last, raw=True)


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = frame.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "trades": "sum",
            "taker_buy_quote": "sum",
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def confirmed_pivots(series: pd.Series, strength: int, kind: str) -> pd.Series:
    """Return pivots at confirmation time, not pivot time, to prevent lookahead."""
    values = series.to_numpy(dtype=float)
    confirmed = np.full(values.size, np.nan)
    for confirmation_i in range(2 * strength, values.size):
        pivot_i = confirmation_i - strength
        window = values[pivot_i - strength : pivot_i + strength + 1]
        if not np.isfinite(window).all():
            continue
        center = values[pivot_i]
        is_pivot = center >= window.max() if kind == "high" else center <= window.min()
        if is_pivot:
            confirmed[confirmation_i] = center
    return pd.Series(confirmed, index=series.index)


def _last_two(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    if len(clean) < 2:
        return np.nan, np.nan
    return float(clean.iloc[-2]), float(clean.iloc[-1])


def enrich_hourly(frame: pd.DataFrame, btc: pd.DataFrame | None, cfg: ScannerConfig) -> pd.DataFrame:
    out = frame.copy()
    atr = wilder_atr(out, cfg.atr_length)
    tr = true_range(out)
    candle_range = (out["high"] - out["low"]).replace(0.0, np.nan)

    out["atr"] = atr
    out["atr_pct"] = _safe_div(atr, out["close"])
    out["tr_atr"] = _safe_div(tr, atr.shift(1))
    out["body_ratio"] = _safe_div((out["close"] - out["open"]).abs(), candle_range)
    out["bull_clv"] = _safe_div(out["close"] - out["low"], candle_range)
    out["rvol_1h"] = _safe_div(out["quote_volume"], out["quote_volume"].shift(1).rolling(168, min_periods=48).median())
    out["quote_volume_24h_roll"] = out["quote_volume"].rolling(24, min_periods=18).sum()
    baseline_24h = out["quote_volume_24h_roll"].shift(24).rolling(24 * 20, min_periods=24 * 5).median()
    out["volume_24h_ratio"] = _safe_div(out["quote_volume_24h_roll"], baseline_24h)

    prior_breakout = out["high"].shift(1).rolling(cfg.breakout_lookback_h, min_periods=24).max()
    macro_high = out["high"].shift(1).rolling(cfg.macro_high_lookback_h, min_periods=120).max()
    out["breakout_level"] = prior_breakout
    out["break_distance_atr"] = _safe_div(out["close"] - prior_breakout, atr)
    out["distance_to_macro_high_atr"] = _safe_div(macro_high - out["close"], atr)
    out["return_24h_calc"] = out["close"].pct_change(24)

    spike = out["rvol_1h"] >= 2.5
    out["prior_spike_count_7d"] = spike.shift(1).rolling(168, min_periods=24).sum()
    spike_origin = out["open"].where(spike).shift(1).ffill(limit=168)
    spike_high = out["high"].where(spike).shift(1).ffill(limit=168)
    impulse_size = (spike_high - spike_origin).abs().replace(0.0, np.nan)
    out["last_spike_retention"] = (out["close"] - spike_origin) / impulse_size

    range_low_7d = out["low"].shift(1).rolling(168, min_periods=48).min()
    range_high_7d = out["high"].shift(1).rolling(168, min_periods=48).max()
    out["range_position_7d"] = _safe_div(out["close"] - range_low_7d, range_high_7d - range_low_7d)
    out["drawdown_from_7d_high"] = out["close"] / range_high_7d - 1.0

    out["taker_buy_ratio_24h"] = _safe_div(
        out["taker_buy_quote"].rolling(24, min_periods=18).sum(),
        out["quote_volume"].rolling(24, min_periods=18).sum(),
    )
    green_quote_volume = out["quote_volume"].where(out["close"] > out["open"], 0.0)
    out["green_volume_share_24h"] = _safe_div(
        green_quote_volume.rolling(24, min_periods=18).sum(),
        out["quote_volume"].rolling(24, min_periods=18).sum(),
    )

    out["pivot_low_confirmed"] = confirmed_pivots(out["low"], cfg.pivot_strength, "low")
    out["pivot_high_confirmed"] = confirmed_pivots(out["high"], cfg.pivot_strength, "high")
    last_pivot_low = out["pivot_low_confirmed"].ffill()
    hypothetical_stop = last_pivot_low - 0.30 * atr
    out["structural_risk_pct"] = (out["close"] - hypothetical_stop) / out["close"]

    h4 = resample_ohlcv(out, "4h")
    h4_atr_pct = _safe_div(wilder_atr(h4, cfg.atr_length), h4["close"])
    mid = h4["close"].rolling(20, min_periods=20).mean()
    std = h4["close"].rolling(20, min_periods=20).std(ddof=0)
    bandwidth = _safe_div(4.0 * std, mid)
    h4["atr_pct_rank"] = percentile_rank_last(h4_atr_pct, 180)
    h4["bandwidth_rank"] = percentile_rank_last(bandwidth, 180)
    h4["pre_compression"] = pd.concat(
        [h4["atr_pct_rank"], h4["bandwidth_rank"]], axis=1
    ).mean(axis=1).rolling(12, min_periods=1).min()
    out["compression_rank"] = h4["pre_compression"].reindex(out.index, method="ffill")

    if btc is not None and not btc.empty:
        btc_close = btc["close"].reindex(out.index).ffill()
        out["btc_return_24h"] = btc_close.pct_change(24, fill_method=None)
        out["btc_return_7d"] = btc_close.pct_change(168, fill_method=None)
        out["relative_strength_24h"] = out["return_24h_calc"] - out["btc_return_24h"]
    else:
        out["btc_return_24h"] = np.nan
        out["btc_return_7d"] = np.nan
        out["relative_strength_24h"] = np.nan

    out["recent_wakeup"] = (
        out["rvol_1h"].shift(1).rolling(cfg.wakeup_lookback_h, min_periods=24).max().ge(2.5)
        | out["volume_24h_ratio"].shift(1).rolling(cfg.wakeup_lookback_h, min_periods=24).max().ge(1.5)
    )
    return out


def evaluate_latest(
    frame: pd.DataFrame,
    btc: pd.DataFrame | None,
    market_row: dict,
    cfg: ScannerConfig,
) -> dict:
    enriched = enrich_hourly(frame, btc, cfg)
    if enriched.empty or len(enriched) < 150:
        raise ValueError("Insufficient hourly history")
    row = enriched.iloc[-1]
    low_prev, low_last = _last_two(enriched["pivot_low_confirmed"])
    higher_low = bool(np.isfinite(low_prev) and np.isfinite(low_last) and low_last > low_prev)

    def clipped(value: float, lo: float, hi: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    near_macro = np.isfinite(row["distance_to_macro_high_atr"]) and row["distance_to_macro_high_atr"] <= 2.0
    break_raw = bool(
        np.isfinite(row["breakout_level"])
        and row["close"] > row["breakout_level"] + cfg.break_buffer_atr * row["atr"]
    )
    expansion_quality = bool(
        row["tr_atr"] >= cfg.min_expansion_tr_atr
        and row["body_ratio"] >= cfg.min_expansion_body_ratio
        and row["bull_clv"] >= cfg.min_expansion_clv
    )
    ignition = bool(break_raw and expansion_quality and row["rvol_1h"] >= cfg.min_ignition_rvol)

    structure_score = (8.0 if higher_low else 0.0) + (7.0 if near_macro else 0.0) + (10.0 if break_raw else 0.0)
    volume_score = 10.0 * clipped(float(row["rvol_1h"]), 1.0, 4.0) + 10.0 * clipped(float(row["volume_24h_ratio"]), 1.0, 3.0)
    comp = float(row["compression_rank"])
    compression_score = 20.0 * float(np.clip((0.60 - comp) / 0.50, 0.0, 1.0)) if np.isfinite(comp) else 0.0
    rs_score = 15.0 * clipped(float(row["relative_strength_24h"]), 0.0, 0.20)
    quality_score = (
        3.5 * clipped(float(row["tr_atr"]), 0.8, 2.5)
        + 3.0 * clipped(float(row["body_ratio"]), 0.35, 0.80)
        + 3.5 * clipped(float(row["bull_clv"]), 0.55, 0.95)
    )
    liquidity_score = 10.0 * clipped(float(market_row["quote_volume_24h"]), cfg.min_quote_volume_24h, 100_000_000.0)
    total_score = structure_score + volume_score + compression_score + rs_score + quality_score + liquidity_score

    chase = bool(
        (np.isfinite(row["break_distance_atr"]) and row["break_distance_atr"] > cfg.no_chase_atr)
        or market_row["return_24h"] > cfg.no_chase_return_24h
    )
    early_liquid_ignition = bool(
        row["quote_volume_24h_roll"] >= 5_000_000.0
        and break_raw
        and expansion_quality
        and row["relative_strength_24h"] >= 0.0
        and 0.05 <= row["return_24h_calc"] <= 0.10
        and 0.05 <= row["structural_risk_pct"] <= 0.10
        and 2.0 <= row["rvol_1h"] <= 10.0
        and 1.5 <= row["volume_24h_ratio"] <= 10.0
    )
    mte_crypto_v0_1 = bool(
        early_liquid_ignition
        and row["quote_volume_24h_roll"] >= 20_000_000.0
        and row["compression_rank"] <= 0.20
        and row["green_volume_share_24h"] >= 0.60
        and row["btc_return_24h"] >= -0.02
    )
    armed_early = bool(
        row["quote_volume_24h_roll"] >= 5_000_000.0
        and bool(row["recent_wakeup"])
        and row["relative_strength_24h"] >= 0.0
        and 0.02 <= row["return_24h_calc"] <= 0.10
        and 0.03 <= row["structural_risk_pct"] <= 0.15
        and 1.5 <= row["rvol_1h"] <= 10.0
        and row["volume_24h_ratio"] >= 1.2
        and row["break_distance_atr"] >= -3.0
    )
    # This is the exact causal candidate gate used to build the frozen
    # MTE_PRE_HUNTER_RF_V0_3 dataset.  Keep it separate from ``armed_early``:
    # the latter is a broader human watch state, while this gate defines the
    # population on which the model's 0.6753 threshold was validated.
    pre_ignition_hunt = bool(
        row["quote_volume_24h_roll"] >= 5_000_000.0
        and bool(row["recent_wakeup"])
        and 1.5 <= row["rvol_1h"] <= 10.0
        and 1.5 <= row["volume_24h_ratio"] <= 10.0
        and -3.0 <= row["break_distance_atr"] <= cfg.break_buffer_atr
        and 0.0 <= row["return_24h_calc"] <= 0.20
        and row["relative_strength_24h"] >= 0.0
    )
    armed_ignition = bool(
        row["quote_volume_24h_roll"] >= 5_000_000.0
        and bool(row["recent_wakeup"])
        and break_raw
        and expansion_quality
        and row["relative_strength_24h"] >= 0.0
        and row["rvol_1h"] >= 2.0
        and row["volume_24h_ratio"] >= 1.5
    )
    speculative_watch = bool(
        1_000_000.0 <= row["quote_volume_24h_roll"] < 5_000_000.0
        and break_raw
        and expansion_quality
        and row["relative_strength_24h"] >= 0.08
        and row["return_24h_calc"] <= 0.25
    )

    if mte_crypto_v0_1 or early_liquid_ignition:
        state = "RESEARCH_IGNITION"
    elif chase and (ignition or bool(row["recent_wakeup"])):
        state = "NO_CHASE"
    elif armed_early:
        state = "ARMED"
    elif speculative_watch:
        state = "SPECULATIVE_WATCH"
    elif total_score >= cfg.watch_score or bool(row["recent_wakeup"]):
        state = "WATCH"
    else:
        state = "IGNORE"

    return {
        "timestamp": enriched.index[-1].isoformat(),
        "symbol": market_row["symbol"],
        "state": state,
        "score": round(total_score, 2),
        "price": float(row["close"]),
        "return_24h": float(market_row["return_24h"]),
        "return_24h_calc": float(row["return_24h_calc"]),
        "quote_volume_24h": float(market_row["quote_volume_24h"]),
        "quote_volume_24h_roll": float(row["quote_volume_24h_roll"]),
        "rvol_1h": float(row["rvol_1h"]),
        "volume_24h_ratio": float(row["volume_24h_ratio"]),
        "relative_strength_24h": float(row["relative_strength_24h"]),
        "btc_return_24h": float(row["btc_return_24h"]),
        "btc_return_7d": float(row["btc_return_7d"]),
        "compression_rank": float(row["compression_rank"]),
        "higher_low": higher_low,
        "recent_wakeup": bool(row["recent_wakeup"]),
        "breakout": break_raw,
        "expansion_quality": expansion_quality,
        "breakout_level": float(row["breakout_level"]),
        "break_distance_atr": float(row["break_distance_atr"]),
        "distance_to_macro_high_atr": float(row["distance_to_macro_high_atr"]),
        "tr_atr": float(row["tr_atr"]),
        "body_ratio": float(row["body_ratio"]),
        "bull_clv": float(row["bull_clv"]),
        "structural_risk_pct": float(row["structural_risk_pct"]),
        "green_volume_share_24h": float(row["green_volume_share_24h"]),
        "taker_buy_ratio_24h": float(row["taker_buy_ratio_24h"]),
        "range_position_7d": float(row["range_position_7d"]),
        "prior_spike_count_7d": float(row["prior_spike_count_7d"]),
        "last_spike_retention": float(row["last_spike_retention"]),
        "early_liquid_ignition": early_liquid_ignition,
        "mte_crypto_v0_1": mte_crypto_v0_1,
        "pre_ignition_hunt": pre_ignition_hunt,
        "armed_early": armed_early,
        "armed_ignition": armed_ignition,
        "config": asdict(cfg),
    }
