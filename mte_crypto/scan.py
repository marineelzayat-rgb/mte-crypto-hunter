from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .binance import BinancePublicClient, liquid_usdt_universe
from .config import DEFAULT_CONFIG, ScannerConfig
from .features import evaluate_latest
from .ranker import REGIME_FEATURES, score as ranker_score
from .regime import build_regime_from_frames


STATE_ORDER = {
    "HUNTER_ALERT": 0,
    "ARMED": 1,
    "RESEARCH_IGNITION": 2,
    "SPECULATIVE_WATCH": 3,
    "WATCH": 4,
    "NO_CHASE": 5,
    "IGNORE": 6,
    "ERROR": 7,
}

PRE_HUNTER_THRESHOLD = 0.6753


def scan_market(
    cfg: ScannerConfig = DEFAULT_CONFIG,
    explicit_symbols: list[str] | None = None,
) -> pd.DataFrame:
    client = BinancePublicClient(cfg.api_base)
    universe = liquid_usdt_universe(
        client,
        min_quote_volume=cfg.min_quote_volume_24h,
        top=cfg.max_universe,
        explicit_symbols=explicit_symbols,
    )
    btc = client.klines("BTCUSDT", cfg.interval, cfg.kline_limit)

    results: list[dict] = []
    frames: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []

    def fetch(market_row: dict) -> tuple[str, pd.DataFrame]:
        return market_row["symbol"], client.klines(market_row["symbol"], cfg.interval, cfg.kline_limit)

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(fetch, row): row for row in universe}
        for future in as_completed(futures):
            market_row = futures[future]
            try:
                symbol, frame = future.result()
                frames[symbol] = frame
            except Exception as exc:  # keep the scan alive when one symbol is malformed
                errors.append(
                    {
                        "symbol": market_row["symbol"],
                        "state": "ERROR",
                        "score": -1.0,
                        "error": str(exc),
                    }
                )

    regime = build_regime_from_frames(frames, cfg.min_quote_volume_24h)
    regime_row = regime.iloc[-1].to_dict() if not regime.empty else {}
    market_by_symbol = {row["symbol"]: row for row in universe}
    for symbol, frame in frames.items():
        try:
            result = evaluate_latest(frame, btc, market_by_symbol[symbol], cfg)
            result.pop("config", None)
            result.update(regime_row)
            results.append(result)
        except Exception as exc:
            errors.append({"symbol": symbol, "state": "ERROR", "score": -1.0, "error": str(exc)})
    results.extend(errors)

    if not results:
        return pd.DataFrame()
    result = pd.DataFrame(results)
    result["hunter_probability"] = np.nan
    model_path = Path(__file__).resolve().parents[1] / "models" / "mte_pre_hunter_rf_v0_3.joblib"
    candidates = result.get("pre_ignition_hunt", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    if model_path.exists() and regime_row and candidates.any():
        model = joblib.load(model_path)
        candidate_scores = ranker_score(model, result.loc[candidates])
        result.loc[candidates, "hunter_probability"] = candidate_scores
        qualified = result.index[candidates & (result["hunter_probability"] >= PRE_HUNTER_THRESHOLD)]
        if len(qualified):
            winner = result.loc[qualified, "hunter_probability"].idxmax()
            result.loc[winner, "state"] = "HUNTER_ALERT"
    result["state_order"] = result["state"].map(STATE_ORDER).fillna(99)
    result = result.sort_values(
        ["state_order", "hunter_probability", "score"], ascending=[True, False, False]
    ).drop(columns="state_order")
    return result.reset_index(drop=True)


def _console_table(frame: pd.DataFrame, show_all: bool) -> str:
    if frame.empty:
        return "No candidates returned."
    visible = frame if show_all else frame[frame["state"] != "IGNORE"]
    cols = [
        "symbol", "state", "score", "return_24h", "rvol_1h",
        "volume_24h_ratio", "relative_strength_24h", "compression_rank",
        "break_distance_atr", "hunter_probability",
    ]
    cols = [c for c in cols if c in visible.columns]
    if visible.empty:
        return "No WATCH/ARMED/PAPER_ENTRY candidates in this scan."
    return visible[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Binance USDT altcoins for MTE Crypto states")
    parser.add_argument("--symbols", nargs="*", help="Optional explicit symbols, e.g. ACEUSDT")
    parser.add_argument("--top", type=int, default=DEFAULT_CONFIG.max_universe)
    parser.add_argument("--min-volume", type=float, default=DEFAULT_CONFIG.min_quote_volume_24h)
    parser.add_argument("--workers", type=int, default=DEFAULT_CONFIG.workers)
    parser.add_argument("--output", default="output/latest_scan.csv")
    parser.add_argument("--json", dest="json_output", default=None)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    cfg = replace(
        DEFAULT_CONFIG,
        max_universe=args.top,
        min_quote_volume_24h=args.min_volume,
        workers=args.workers,
    )
    frame = scan_market(cfg, explicit_symbols=args.symbols)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(frame.to_dict(orient="records"), indent=2, default=str))

    print(_console_table(frame, args.show_all))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
