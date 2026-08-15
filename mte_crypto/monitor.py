from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG
from .scan import scan_market


ALERT_STATES = {"HUNTER_ALERT"}


def format_alert(row: dict) -> str:
    state = row.get("state", "-")
    action = {
        "HUNTER_ALERT": "High-priority explosion candidate — discovery only, no order",
    }.get(state, "Observe")
    return (
        f"MTE CRYPTO — {state}\n"
        f"{row['symbol']} @ {row.get('price', 0):.8g}\n"
        f"24h return: {100 * row.get('return_24h', 0):.1f}%\n"
        f"24h quote volume: ${row.get('quote_volume_24h', 0):,.0f}\n"
        f"H1 RVOL: {row.get('rvol_1h', 0):.2f}x\n"
        f"24h volume ratio: {row.get('volume_24h_ratio', 0):.2f}x\n"
        f"Relative strength vs BTC: {100 * row.get('relative_strength_24h', 0):.1f}%\n"
        f"Structural risk: {100 * row.get('structural_risk_pct', 0):.1f}%\n"
        f"Break distance: {row.get('break_distance_atr', 0):.2f} ATR\n"
        f"Action: {action}"
    )


def send_telegram(message: str) -> bool:
    token = os.environ.get("MTE_TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("MTE_TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return 200 <= response.status < 300


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def run_once(output_dir: Path, top: int) -> list[str]:
    cfg = replace(DEFAULT_CONFIG, max_universe=top)
    scan = scan_market(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan.to_csv(output_dir / "latest_scan.csv", index=False)

    state_path = output_dir / "monitor_state.json"
    old_state = load_state(state_path)
    new_state: dict[str, dict] = {}
    alerts: list[str] = []
    for row in scan.to_dict(orient="records"):
        symbol = row["symbol"]
        current = row["state"]
        previous = old_state.get(symbol, {}).get("state")
        new_state[symbol] = {"state": current, "timestamp": row.get("timestamp")}
        if current in ALERT_STATES and current != previous:
            message = format_alert(row)
            alerts.append(message)
            sent = send_telegram(message)
            if not sent:
                print("\n" + message + "\n")
    state_path.write_text(json.dumps(new_state, indent=2))
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MTE Crypto scanner continuously")
    parser.add_argument("--top", type=int, default=DEFAULT_CONFIG.max_universe)
    parser.add_argument("--output-dir", default="output/live")
    parser.add_argument("--every-minutes", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    while True:
        started = time.time()
        alerts = run_once(output, args.top)
        print(f"Scan complete: {len(alerts)} new state transitions")
        if args.once:
            break
        elapsed = time.time() - started
        time.sleep(max(args.every_minutes * 60 - elapsed, 5))


if __name__ == "__main__":
    main()
