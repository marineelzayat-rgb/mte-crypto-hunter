from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import urlopen

from mte_crypto.alert_store import (
    backfill_legacy_outcomes,
    bootstrap_active_alerts,
    record_alert,
    seed_legacy_alerts,
    status_payload,
    update_alert_outcomes,
)
from mte_crypto.status_server import start_status_server


UTC = timezone.utc


class AlertStoreTests(unittest.TestCase):
    def test_legacy_alerts_are_seeded_once_and_backfilled_from_candles(self):
        class FakeClient:
            def kline_rows(self, symbol, **kwargs):
                start = kwargs["start_time_ms"]
                return [
                    [start, "1", "0.160", "0.140", "0.150"],
                    [start + 6 * 3_600_000, "1", "0.155", "0.130", "0.135"],
                ]

        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            self.assertEqual(seed_legacy_alerts(data_dir), 2)
            self.assertEqual(seed_legacy_alerts(data_dir), 0)
            restored = backfill_legacy_outcomes(
                data_dir,
                FakeClient(),
                now=datetime(2026, 8, 16, 3, 0, tzinfo=UTC),
            )
            self.assertEqual(restored, 2)
            cow = next(
                item
                for item in status_payload(data_dir)["alerts"]
                if item["symbol"] == "COWUSDT"
            )
            self.assertAlmostEqual(cow["outcome"]["max_return"], 0.160 / 0.1465 - 1)
            self.assertAlmostEqual(cow["outcome"]["min_return"], 0.130 / 0.1465 - 1)
            self.assertIn("6h", cow["outcome"]["checkpoints"])

    def test_public_ledger_tracks_outcomes_without_leaking_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            observed = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
            record_alert(
                data_dir,
                {
                    "symbol": "TESTUSDT",
                    "state": "EARLY_PULSE",
                    "price": 2.0,
                    "return_24h": 0.12,
                    "api_key": "should-never-appear",
                    "MTE_TELEGRAM_BOT_TOKEN": "also-secret",
                },
                source="pulse",
                observed_at=observed,
            )
            update_alert_outcomes(
                data_dir,
                [{"symbol": "TESTUSDT", "last_price": 2.5}],
                now=observed + timedelta(hours=6),
            )

            serialized = json.dumps(status_payload(data_dir))
            self.assertIn("TESTUSDT", serialized)
            self.assertNotIn("should-never-appear", serialized)
            self.assertNotIn("also-secret", serialized)
            outcome = status_payload(data_dir)["alerts"][0]["outcome"]
            self.assertAlmostEqual(outcome["current_return"], 0.25)
            self.assertAlmostEqual(outcome["checkpoints"]["6h"]["return"], 0.25)

    def test_bootstrap_and_http_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            detected = datetime(2026, 8, 16, 13, 0, tzinfo=UTC).isoformat()
            (data_dir / "active_candidates.json").write_text(
                json.dumps(
                    {
                        "NILUSDT": {
                            "detected_at": detected,
                            "expires_at": detected,
                            "price": 0.04926,
                            "hunter_probability": 0.81,
                            "private_note": "hidden",
                        }
                    }
                )
            )
            bootstrap_active_alerts(data_dir)
            payload = status_payload(data_dir)
            self.assertEqual(payload["alerts"][0]["symbol"], "NILUSDT")
            self.assertNotIn("private_note", payload["active"]["hunter"]["NILUSDT"])

            server = start_status_server(data_dir, 0)
            try:
                port = server.server_address[1]
                with urlopen(f"http://127.0.0.1:{port}/health") as response:
                    self.assertEqual(json.load(response), {"status": "ok"})
                with urlopen(f"http://127.0.0.1:{port}/status.json") as response:
                    self.assertEqual(json.load(response)["alerts"][0]["symbol"], "NILUSDT")
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
