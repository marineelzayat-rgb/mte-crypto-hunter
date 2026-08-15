from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .config import DEFAULT_CONFIG
from .features import enrich_hourly
from .research import label_explosions, load_csv
from .rule_eval import rules


RANK_FEATURES = [
    "rvol_1h",
    "volume_24h_ratio",
    "relative_strength_24h",
    "range_position_7d",
    "last_spike_retention",
    "prior_spike_count_7d",
    "green_volume_share_24h",
    "tr_atr",
]


def build_candidates(data_dir: Path, btc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    btc = load_csv(btc_path)
    candidates: list[pd.DataFrame] = []
    events: list[dict] = []
    keep_rules = {"MTE_STRICT", "ARMED_IGNITION", "MULTI_SPIKE_IGNITION", "REIGNITION_BREAK"}

    for path in sorted(data_dir.glob("*.csv")):
        if path.resolve() == btc_path.resolve() or path.stat().st_size == 0:
            continue
        try:
            raw = load_csv(path)
        except (EmptyDataError, ValueError):
            continue
        if len(raw) < 300:
            continue
        frame = label_explosions(enrich_hourly(raw, btc, DEFAULT_CONFIG))
        symbol = path.stem.upper()
        for timestamp in frame.index[frame["event_start"]]:
            events.append({"symbol": symbol, "event_time": timestamp})
        for name, mask in rules(frame).items():
            if name not in keep_rules:
                continue
            selected = frame.loc[mask.fillna(False), RANK_FEATURES + ["return_24h_calc", "close"]].copy()
            if selected.empty:
                continue
            selected["symbol"] = symbol
            selected["rule"] = name
            selected["signal_time"] = selected.index
            candidates.append(selected.reset_index(drop=True))
    return (pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(), pd.DataFrame(events))


def rank_candidates(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked: list[pd.DataFrame] = []
    for (rule, timestamp), group in candidates.groupby(["rule", "signal_time"], sort=False):
        work = group.copy()
        score = pd.Series(0.0, index=work.index)
        for feature in RANK_FEATURES:
            values = work[feature].replace([np.inf, -np.inf], np.nan)
            score += values.rank(pct=True).fillna(0.5)
        # Penalize entries already in a terminal-looking 24h move while not
        # forbidding moderate early momentum.
        chase_penalty = np.clip((work["return_24h_calc"].fillna(0.0) - 0.30) / 0.40, 0.0, 1.0)
        work["rank_score"] = score / len(RANK_FEATURES) - 0.25 * chase_penalty
        ranked.append(work.nlargest(top_k, "rank_score"))
    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


def cluster_and_evaluate(ranked: pd.DataFrame, events: pd.DataFrame, cooldown_h: int = 72) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    kept_rows: list[pd.DataFrame] = []
    for rule, rule_rows in ranked.groupby("rule"):
        clustered_parts = []
        for symbol, group in rule_rows.sort_values("signal_time").groupby("symbol"):
            keep = []
            last = None
            for timestamp in group["signal_time"]:
                if last is None or (timestamp - last).total_seconds() >= cooldown_h * 3600:
                    keep.append(True)
                    last = timestamp
                else:
                    keep.append(False)
            clustered_parts.append(group.loc[keep])
        signals = pd.concat(clustered_parts) if clustered_parts else rule_rows.iloc[0:0]
        true_flags = []
        leads = []
        captured = 0
        rule_events = events
        for _, signal in signals.iterrows():
            matches = rule_events[
                (rule_events["symbol"] == signal["symbol"])
                & (rule_events["event_time"] > signal["signal_time"])
                & (rule_events["event_time"] <= signal["signal_time"] + pd.Timedelta(hours=48))
            ]
            true_flags.append(not matches.empty)
            leads.append((matches["event_time"].iloc[0] - signal["signal_time"]).total_seconds() / 3600 if not matches.empty else np.nan)
        signals = signals.copy()
        signals["true_within_48h"] = true_flags
        signals["lead_h"] = leads
        kept_rows.append(signals)

        for _, event in rule_events.iterrows():
            if ((signals["symbol"] == event["symbol"]) & (signals["signal_time"] >= event["event_time"] - pd.Timedelta(hours=48)) & (signals["signal_time"] < event["event_time"])).any():
                captured += 1
        summaries.append(
            {
                "rule": rule,
                "signals": len(signals),
                "true_signals": int(np.sum(true_flags)),
                "precision": float(np.mean(true_flags)) if true_flags else np.nan,
                "events": len(rule_events),
                "captured_events": captured,
                "recall": captured / len(rule_events) if len(rule_events) else np.nan,
                "median_lead_h": float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan,
            }
        )
    return pd.DataFrame(summaries).sort_values("precision", ascending=False), pd.concat(kept_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-sectional top-K MTE Crypto candidates")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--btc", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="output/ranked_evaluation.csv")
    args = parser.parse_args()

    candidates, events = build_candidates(Path(args.data_dir), Path(args.btc))
    ranked = rank_candidates(candidates, args.top_k)
    summary, signals = cluster_and_evaluate(ranked, events)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    signals.to_csv(output.with_name(output.stem + "_signals.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()

