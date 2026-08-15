from datetime import datetime, timedelta, timezone
import unittest

from mte_crypto.pulse import (
    PulseConfig,
    append_history,
    evaluate_pulses,
    update_pulse_candidates,
)


UTC = timezone.utc


class PulseTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 15, 12, 10, tzinfo=UTC)
        self.history = {
            "COWUSDT": [
                {
                    "at": (self.now - timedelta(minutes=10)).isoformat(),
                    "price": 1.0,
                    "quote_volume_24h": 100_000.0,
                },
                {
                    "at": (self.now - timedelta(minutes=5)).isoformat(),
                    "price": 1.005,
                    "quote_volume_24h": 110_000.0,
                },
            ]
        }

    def test_cold_market_wakeup_is_detected_below_one_million_volume(self):
        rows = evaluate_pulses(
            [
                {
                    "symbol": "COWUSDT",
                    "last_price": 1.055,
                    "return_24h": 0.08,
                    "quote_volume_24h": 250_000.0,
                }
            ],
            self.history,
            now=self.now,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "EARLY_PULSE")
        self.assertGreater(rows[0]["return_5m"], 0.04)
        self.assertGreater(rows[0]["volume_acceleration"], 3.0)

    def test_extended_move_is_recorded_but_marked_no_chase(self):
        rows = evaluate_pulses(
            [
                {
                    "symbol": "COWUSDT",
                    "last_price": 1.055,
                    "return_24h": 0.36,
                    "quote_volume_24h": 250_000.0,
                }
            ],
            self.history,
            now=self.now,
        )
        self.assertEqual(rows[0]["state"], "RAPID_MOVE_NO_CHASE")

    def test_price_move_without_new_volume_is_ignored(self):
        rows = evaluate_pulses(
            [
                {
                    "symbol": "COWUSDT",
                    "last_price": 1.055,
                    "return_24h": 0.08,
                    "quote_volume_24h": 120_000.0,
                }
            ],
            self.history,
            now=self.now,
        )
        self.assertEqual(rows, [])

    def test_active_candidate_alert_is_deduplicated(self):
        row = {"symbol": "COWUSDT", "state": "EARLY_PULSE", "price": 1.055}
        active, alerts = update_pulse_candidates(
            [row], {}, now=self.now, ttl_minutes=120
        )
        self.assertEqual(len(alerts), 1)
        _, alerts = update_pulse_candidates(
            [row], active, now=self.now + timedelta(minutes=5), ttl_minutes=120
        )
        self.assertEqual(alerts, [])

    def test_history_is_bounded(self):
        old = self.now - timedelta(minutes=120)
        history = {
            "COWUSDT": [
                {"at": old.isoformat(), "price": 0.9, "quote_volume_24h": 1.0}
            ]
        }
        updated = append_history(
            [
                {
                    "symbol": "COWUSDT",
                    "last_price": 1.0,
                    "quote_volume_24h": 2.0,
                }
            ],
            history,
            now=self.now,
            history_minutes=90,
        )
        self.assertEqual(len(updated["COWUSDT"]), 1)


if __name__ == "__main__":
    unittest.main()
