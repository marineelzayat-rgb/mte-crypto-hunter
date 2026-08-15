from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def activation_flow_features(frame: pd.DataFrame, position: int) -> dict[str, float]:
    """Causal trade-flow features available at a completed H1 activation bar.

    Binance klines include quote volume, trade count and taker-buy quote
    volume.  No future row is referenced here.  These features belong to the
    MTE activation timestamp, rather than the earlier Hunter timestamp.
    """
    if position < 0 or position >= len(frame):
        raise IndexError("position is outside frame")
    required = {"open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing flow columns: {sorted(missing)}")

    quote = pd.to_numeric(frame["quote_volume"], errors="coerce")
    taker_buy = pd.to_numeric(frame["taker_buy_quote"], errors="coerce")
    trades = pd.to_numeric(frame["trades"], errors="coerce")
    prior_quote_median = quote.shift(1).rolling(168, min_periods=48).median().iloc[position]
    prior_trades_median = trades.shift(1).rolling(168, min_periods=48).median().iloc[position]

    row = frame.iloc[position]
    candle_range = float(row["high"] - row["low"])
    features: dict[str, float] = {
        "flow_rvol_1h": _ratio(float(quote.iloc[position]), float(prior_quote_median)),
        "flow_trade_intensity_1h": _ratio(float(trades.iloc[position]), float(prior_trades_median)),
        "flow_close_location": _ratio(float(row["close"] - row["low"]), candle_range),
        "flow_bar_return": _ratio(float(row["close"]), float(row["open"])) - 1.0,
    }
    signed_quote = 2.0 * taker_buy - quote
    for hours in (1, 3, 6, 24):
        start = max(0, position - hours + 1)
        q = float(quote.iloc[start : position + 1].sum())
        buy = float(taker_buy.iloc[start : position + 1].sum())
        features[f"flow_taker_buy_ratio_{hours}h"] = _ratio(buy, q)
        features[f"flow_delta_ratio_{hours}h"] = _ratio(2.0 * buy - q, q)

    start6 = max(0, position - 5)
    features["flow_positive_delta_share_6h"] = float(
        (signed_quote.iloc[start6 : position + 1] > 0).mean()
    )
    features["flow_high_volume_share_6h"] = float(
        (quote.iloc[start6 : position + 1] > 1.5 * prior_quote_median).mean()
    )
    return features


def order_book_features(
    bids: Sequence[Sequence[float | str]],
    asks: Sequence[Sequence[float | str]],
    bands_bps: tuple[int, ...] = (10, 25, 50),
) -> dict[str, float]:
    """Execution and imbalance metrics from one visible order-book snapshot.

    A snapshot is diagnostic only; persistence must be evaluated across many
    snapshots before it can confirm or veto an MTE ignition.
    """
    bid_rows = sorted(((float(p), float(q)) for p, q in bids if float(q) > 0), reverse=True)
    ask_rows = sorted((float(p), float(q)) for p, q in asks if float(q) > 0)
    if not bid_rows or not ask_rows:
        raise ValueError("both bids and asks are required")
    best_bid, best_bid_qty = bid_rows[0]
    best_ask, best_ask_qty = ask_rows[0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000.0
    microprice = (best_ask * best_bid_qty + best_bid * best_ask_qty) / (best_bid_qty + best_ask_qty)
    out = {
        "book_mid": mid,
        "book_spread_bps": spread_bps,
        "book_microprice_edge_bps": (microprice - mid) / mid * 10_000.0,
        "book_top_imbalance": _ratio(best_bid_qty - best_ask_qty, best_bid_qty + best_ask_qty),
    }
    for band in bands_bps:
        width = band / 10_000.0
        bid_quote = sum(price * qty for price, qty in bid_rows if price >= mid * (1.0 - width))
        ask_quote = sum(price * qty for price, qty in ask_rows if price <= mid * (1.0 + width))
        out[f"book_bid_depth_{band}bps"] = float(bid_quote)
        out[f"book_ask_depth_{band}bps"] = float(ask_quote)
        out[f"book_imbalance_{band}bps"] = _ratio(bid_quote - ask_quote, bid_quote + ask_quote)
    return out
