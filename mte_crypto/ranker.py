from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from .dataset import COIN_FEATURES, REGIME_FEATURES


FEATURES = COIN_FEATURES + REGIME_FEATURES
DEFAULT_THRESHOLD = 0.75


def train(paths: list[Path]):
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    x = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = data["target_explosion_48h"].astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    )
    model.fit(x, y)
    return model, data


def score(model, data: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(data[FEATURES].replace([np.inf, -np.inf], np.nan))[:, 1]


def select_candidates(model, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    data = data.copy()
    data["hunter_probability"] = score(model, data)
    selected = data[data["hunter_probability"] >= threshold].copy()
    selected = selected.sort_values(
        ["signal_time", "hunter_probability"], ascending=[True, False]
    ).groupby("signal_time").head(1)
    keep: list[bool] = []
    last: dict[str, pd.Timestamp] = {}
    for _, row in selected.sort_values("signal_time").iterrows():
        ok = row["symbol"] not in last or row["signal_time"] - last[row["symbol"]] >= pd.Timedelta(hours=72)
        keep.append(ok)
        if ok:
            last[row["symbol"]] = row["signal_time"]
    return selected.loc[keep]


def metrics(model, path: Path, threshold: float) -> dict:
    data = pd.read_csv(path, parse_dates=["signal_time"])
    probabilities = score(model, data)
    selected = select_candidates(model, data, threshold)
    target = data["target_explosion_48h"].astype(int)
    return {
        "dataset": str(path),
        "rows": len(data),
        "base_rate": float(target.mean()),
        "average_precision": float(average_precision_score(target, probabilities)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "threshold": threshold,
        "selected": len(selected),
        "selected_precision": float(selected["target_explosion_48h"].mean()) if len(selected) else None,
        "selected_true": int(selected["target_explosion_48h"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen MTE explosion hunter ranker")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--evaluate", nargs="*")
    parser.add_argument("--model", default="models/mte_hunter_rf_v0_2.joblib")
    parser.add_argument("--metadata", default="models/mte_hunter_rf_v0_2.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--export-selected-dir")
    parser.add_argument("--version", default="MTE_HUNTER_RF_V0_2")
    args = parser.parse_args()
    model, training = train([Path(path) for path in args.train])
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report = {
        "version": args.version,
        "purpose": "candidate discovery only; not an entry or order model",
        "features": FEATURES,
        "threshold": args.threshold,
        "training_rows": len(training),
        "training_base_rate": float(training["target_explosion_48h"].mean()),
        "evaluations": [metrics(model, Path(path), args.threshold) for path in (args.evaluate or [])],
    }
    metadata = Path(args.metadata)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(report, indent=2))
    if args.export_selected_dir:
        selected_dir = Path(args.export_selected_dir)
        selected_dir.mkdir(parents=True, exist_ok=True)
        for path_text in args.evaluate or []:
            path = Path(path_text)
            data = pd.read_csv(path, parse_dates=["signal_time"])
            selected = select_candidates(model, data, args.threshold)
            selected.to_csv(selected_dir / f"{path.stem}_selected.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
