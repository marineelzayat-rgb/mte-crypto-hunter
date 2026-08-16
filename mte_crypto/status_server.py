from __future__ import annotations

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

from .alert_store import status_payload


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
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30"><title>MTE Crypto Hunter Status</title>
<style>
:root{{--bg:#090b10;--card:#111622;--line:#273044;--text:#eef2ff;--muted:#9aa5bd;--green:#3ddc97}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:22px}}h1{{margin:0 0 6px}}p{{color:var(--muted)}}.pill{{display:inline-block;padding:6px 10px;border:1px solid var(--line);border-radius:99px;margin:4px}}
.wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--card);margin-top:18px}}
table{{border-collapse:collapse;width:100%;min-width:1250px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);position:sticky;top:0;background:var(--card)}}
.ok{{color:var(--green)}}a{{color:#8ab4ff}}</style></head>
<body><main><h1>MTE Crypto Hunter</h1><p class="ok">Live research status · refreshes every 30 seconds</p>
<p>Discovery research only. No order placement. Times are UTC.</p>
<div><span class="pill">Hunter active: {hunter_count}</span>
<span class="pill">Pulse active: {pulse_count}</span>
<span class="pill"><a href="/status.json">JSON</a></span></div>
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
                payload = json.dumps(status_payload(data_dir), separators=(",", ":")).encode()
                self._send(payload, "application/json")
                return
            if path in {"/", "/status"}:
                html = render_status_html(status_payload(data_dir)).encode()
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
