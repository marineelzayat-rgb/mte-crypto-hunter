from __future__ import annotations

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

from .alert_store import status_payload
from .futures_shadow import futures_shadow_payload
from .live_spot import live_spot_payload
from .paper_portfolio import paper_portfolio_payload


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _full_status_payload(data_dir: Path) -> dict:
    payload = status_payload(data_dir)
    payload["paper_portfolio"] = paper_portfolio_payload(data_dir)
    payload["futures_shadow"] = futures_shadow_payload(data_dir)
    payload["market_regime"] = _read_json(data_dir / "market_regime.json")
    payload["live_spot"] = live_spot_payload(data_dir)
    return payload


def _percent(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{100 * float(value):+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _price(value) -> str:
    try:
        return f"{float(value or 0):.8g}"
    except (TypeError, ValueError):
        return "—"


def _money(value) -> str:
    try:
        return f"${float(value or 0):,.2f}"
    except (TypeError, ValueError):
        return "—"


def render_status_html(payload: dict) -> str:
    rows = []
    for alert in payload.get("alerts", []):
        outcome = alert.get("outcome") or {}
        checkpoints = outcome.get("checkpoints") or {}
        rows.append(
            "<tr>"
            f"<td>{escape(str(alert.get('observed_at', '—'))[:19].replace('T', ' '))} UTC</td>"
            f"<td><strong>{escape(str(alert.get('symbol', '—')))}</strong></td>"
            f"<td>{escape(str(alert.get('state', '—')))}</td>"
            f"<td>{escape(str(alert.get('source', '—')))}</td>"
            f"<td>{escape(_price(alert.get('price')))}</td>"
            f"<td>{_percent(alert.get('return_24h'))}</td>"
            f"<td>{_percent(outcome.get('current_return'))}</td>"
            f"<td>{_percent(outcome.get('max_return'))}</td>"
            f"<td>{_percent(outcome.get('min_return'))}</td>"
            f"<td>{_percent((checkpoints.get('6h') or {}).get('return'))}</td>"
            f"<td>{_percent((checkpoints.get('12h') or {}).get('return'))}</td>"
            f"<td>{_percent((checkpoints.get('24h') or {}).get('return'))}</td>"
            f"<td>{_percent((checkpoints.get('48h') or {}).get('return'))}</td>"
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="13">No alerts recorded yet.</td></tr>'
    active = payload.get("active") or {}
    hunter_count = len(active.get("hunter") or {})
    pulse_count = len(active.get("pulse") or {})
    paper = payload.get("paper_portfolio") or {}
    futures = payload.get("futures_shadow") or {}
    regime = payload.get("market_regime") or {}
    live = payload.get("live_spot") or {}
    live_connection = live.get("connection") or {}
    position_rows = []
    for position in paper.get("open_positions", []):
        position_rows.append(
            "<tr>"
            f"<td>{escape(str(position.get('slot_id', '—')))}</td>"
            f"<td><strong>{escape(str(position.get('symbol', '—')))}</strong></td>"
            f"<td>{escape(str(position.get('opened_at', '—'))[:19].replace('T', ' '))} UTC</td>"
            f"<td>{escape(_price(position.get('entry_price')))}</td>"
            f"<td>{escape(_price(position.get('current_price')))}</td>"
            f"<td>{escape(_price(position.get('stop_price')))}</td>"
            f"<td>{'ACTIVE' if position.get('trail_active') else 'WAITING +5%'}</td>"
            f"<td>{_percent(position.get('current_return'))}</td>"
            f"<td>{escape(_money(position.get('unrealized_pnl')))}</td>"
            "</tr>"
        )
    positions_body = "".join(position_rows) or '<tr><td colspan="9">No paper positions open.</td></tr>'
    closed_rows = []
    for trade in (paper.get("closed_trades") or [])[:20]:
        closed_rows.append(
            "<tr>"
            f"<td>{escape(str(trade.get('slot_id', '—')))}</td>"
            f"<td><strong>{escape(str(trade.get('symbol', '—')))}</strong></td>"
            f"<td>{escape(str(trade.get('closed_at', '—'))[:19].replace('T', ' '))} UTC</td>"
            f"<td>{escape(str(trade.get('exit_reason', '—')))}</td>"
            f"<td>{escape(_price(trade.get('entry_price')))}</td>"
            f"<td>{escape(_price(trade.get('exit_price')))}</td>"
            f"<td>{_percent(trade.get('return'))}</td>"
            f"<td>{escape(_money(trade.get('pnl')))}</td>"
            "</tr>"
        )
    closed_body = "".join(closed_rows) or '<tr><td colspan="8">No paper trades closed yet.</td></tr>'
    futures_rows = []
    for position in futures.get("open_positions", []):
        futures_rows.append(
            "<tr>"
            f"<td>{escape(str(position.get('slot_id', '—')))}</td>"
            f"<td><strong>{escape(str(position.get('symbol', '—')))}</strong></td>"
            f"<td>{escape(_price(position.get('entry_price')))}</td>"
            f"<td>{escape(_price(position.get('current_bid')))}</td>"
            f"<td>{_percent(position.get('entry_spread'))}</td>"
            f"<td>{_percent(position.get('last_funding_rate'))}</td>"
            f"<td>{_percent(position.get('current_return'))}</td>"
            f"<td>{escape(_money(position.get('unrealized_pnl')))}</td>"
            "</tr>"
        )
    futures_body = "".join(futures_rows) or '<tr><td colspan="8">No 2x shadow positions open.</td></tr>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30"><title>MTE Crypto Hunter Status</title>
<style>
:root{{--bg:#090b10;--card:#111622;--line:#273044;--text:#eef2ff;--muted:#9aa5bd;--green:#3ddc97}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:22px}}h1{{margin:0 0 6px}}p{{color:var(--muted)}}.pill{{display:inline-block;padding:6px 10px;border:1px solid var(--line);border-radius:99px;margin:4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:18px}}.card{{padding:14px;border:1px solid var(--line);border-radius:14px;background:var(--card)}}.label{{color:var(--muted);font-size:12px}}.value{{font-size:22px;font-weight:700;margin-top:4px}}
.wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--card);margin-top:18px}}
table{{border-collapse:collapse;width:100%;min-width:1250px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);position:sticky;top:0;background:var(--card)}}
.ok{{color:var(--green)}}a{{color:#8ab4ff}}</style></head>
<body><main><h1>MTE Crypto Hunter</h1><p class="ok">Live research status · refreshes every 30 seconds</p>
<p>Paper research remains active. Real Spot is separately gated and defaults to SAFE_DISABLED. Times are UTC.</p>
<div><span class="pill">Hunter active: {hunter_count}</span>
<span class="pill">Pulse active: {pulse_count}</span>
<span class="pill">Regime: {escape(str(regime.get('state', 'UNKNOWN')))}</span>
<span class="pill">Live Spot: {escape(str(live.get('mode', 'SAFE_DISABLED')))}</span>
<span class="pill">Binance link: {'CONNECTED' if live_connection.get('connected') else 'NOT CONNECTED'}</span>
<span class="pill"><a href="/status.json">JSON</a></span></div>
<h2>Wave Rider paper portfolio</h2>
<p>Starts at $100 · 16 isolated slots · EARLY_PULSE plus controlled BULL_CONTINUATION · 7.5% initial stop · staged profit floors · 3.5 ATR trail (4.5 in bull mode) · activated runners may continue for 7 days.</p>
<div class="cards">
<div class="card"><div class="label">Paper equity</div><div class="value">{escape(_money(paper.get('equity')))}</div></div>
<div class="card"><div class="label">Total return</div><div class="value">{_percent(paper.get('total_return'))}</div></div>
<div class="card"><div class="label">Open positions</div><div class="value">{paper.get('open_count', 0)} / {paper.get('max_positions', 16)}</div></div>
<div class="card"><div class="label">Available slots</div><div class="value">{paper.get('available_slots', 16)}</div></div>
<div class="card"><div class="label">Realized P&amp;L</div><div class="value">{escape(_money(paper.get('realized_pnl')))}</div></div>
</div>
<div class="wrap"><table><thead><tr><th>Slot</th><th>Symbol</th><th>Opened</th><th>Entry</th><th>Current</th><th>Stop</th><th>Trail</th><th>Return</th><th>Unrealized P&amp;L</th></tr></thead>
<tbody>{positions_body}</tbody></table></div>
<h2>Recent paper exits</h2>
<div class="wrap"><table><thead><tr><th>Slot</th><th>Symbol</th><th>Closed</th><th>Reason</th><th>Entry</th><th>Exit</th><th>Return</th><th>P&amp;L</th></tr></thead>
<tbody>{closed_body}</tbody></table></div>
<h2>USD-M Futures 2x shadow</h2>
<p>Mirrors the same accepted spot entries and exits · long 2x · executable ask/bid prices · estimated taker fees and observed funding · PAPER ONLY, no orders.</p>
<div class="cards">
<div class="card"><div class="label">2x shadow equity</div><div class="value">{escape(_money(futures.get('equity')))}</div></div>
<div class="card"><div class="label">2x total return</div><div class="value">{_percent(futures.get('total_return'))}</div></div>
<div class="card"><div class="label">Open positions</div><div class="value">{futures.get('open_count', 0)} / {futures.get('max_positions', 16)}</div></div>
<div class="card"><div class="label">Estimated fees</div><div class="value">{escape(_money(futures.get('fees')))}</div></div>
<div class="card"><div class="label">Funding P&amp;L</div><div class="value">{escape(_money(futures.get('funding_pnl')))}</div></div>
</div>
<div class="wrap"><table><thead><tr><th>Slot</th><th>Symbol</th><th>Entry ask</th><th>Current bid</th><th>Entry spread</th><th>Funding rate</th><th>Return</th><th>Unrealized P&amp;L</th></tr></thead>
<tbody>{futures_body}</tbody></table></div>
<h2>Discovery ledger</h2>
<div class="wrap"><table><thead><tr><th>Detected</th><th>Symbol</th><th>State</th><th>Source</th><th>Price</th><th>24h at alert</th><th>Current</th><th>Max</th><th>Min</th><th>6h</th><th>12h</th><th>24h</th><th>48h</th></tr></thead>
<tbody>{body}</tbody></table></div></main></body></html>"""


def make_handler(data_dir: Path):
    class StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/health":
                self._send(b'{"status":"ok"}', "application/json")
                return
            if path == "/status.json":
                payload = json.dumps(_full_status_payload(data_dir), separators=(",", ":")).encode()
                self._send(payload, "application/json")
                return
            if path in {"/", "/status"}:
                html = render_status_html(_full_status_payload(data_dir)).encode()
                self._send(html, "text/html; charset=utf-8")
                return
            self.send_error(404)

        def _send(self, payload: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    return StatusHandler


def start_status_server(data_dir: Path, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(data_dir))
    thread = threading.Thread(target=server.serve_forever, name="mte-status", daemon=True)
    thread.start()
    return server
