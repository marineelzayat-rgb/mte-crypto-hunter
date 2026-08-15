from datetime import datetime, timezone
import unittest

from mte_crypto.daemon import update_active_candidates


class ActiveCandidateTests(unittest.TestCase):
    def test_alert_is_added_once_and_expires_after_ttl(self):
        now = datetime(2026, 8, 15, 1, tzinfo=timezone.utc)
        rows = [{"symbol": "ACEUSDT", "state": "HUNTER_ALERT", "price": 0.1}]
        active, states, alerts = update_active_candidates(
            rows, {}, {}, now=now, ttl_hours=48
        )
        self.assertEqual(list(active), ["ACEUSDT"])
        self.assertEqual(len(alerts), 1)

        active, states, alerts = update_active_candidates(
            rows, active, states, now=now, ttl_hours=48
        )
        self.assertEqual(len(alerts), 0)


if __name__ == "__main__":
    unittest.main()
