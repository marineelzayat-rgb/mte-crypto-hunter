from datetime import datetime, timezone
from pathlib import Path
import gzip
import json
import tempfile
import unittest

from mte_crypto.book_collector import (
    collection_timeout_seconds,
    reclaim_order_book_storage,
)
from mte_crypto.daemon import (
    build_order_book_collection_plan,
    update_active_candidates,
    watchdog_is_stale,
)
from mte_crypto.paper_portfolio import open_paper_position


class ActiveCandidateTests(unittest.TestCase):
    def test_order_book_storage_is_compacted_to_mergeable_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            raw = data_dir / "order_book" / "2026-08-20.jsonl.gz"
            raw.parent.mkdir(parents=True)
            with gzip.open(raw, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"symbol": "TESTUSDT", "book_spread_bps": 2.0}) + "\n")
                handle.write(json.dumps({"symbol": "TESTUSDT", "book_spread_bps": 4.0}) + "\n")
            compacted = reclaim_order_book_storage(
                data_dir, min_free_bytes=10**18, max_raw_bytes=0
            )
            self.assertEqual(compacted, [raw.name])
            self.assertFalse(raw.exists())
            summary = json.loads(
                (data_dir / "order_book_summary" / "2026-08-20.json").read_text()
            )
            metric = summary["symbols"]["TESTUSDT"]["metrics"]["book_spread_bps"]
            self.assertEqual(metric["count"], 2)
            self.assertEqual(metric["min"], 2.0)
            self.assertEqual(metric["max"], 4.0)
            self.assertEqual(metric["sum"], 6.0)

    def test_watchdog_detects_only_expired_heartbeat(self):
        self.assertFalse(watchdog_is_stale(100.0, 699.0, 600.0))
        self.assertTrue(watchdog_is_stale(100.0, 701.0, 600.0))

    def test_order_book_deadline_is_absolute_and_bounded(self):
        self.assertEqual(collection_timeout_seconds(None), None)
        self.assertAlmostEqual(collection_timeout_seconds(1.0), 1.25)
        self.assertAlmostEqual(collection_timeout_seconds(120.0), 130.0)

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

    def test_order_book_plan_keeps_open_trades_and_uses_adaptive_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            now = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)
            open_paper_position(
                data_dir,
                {
                    "id": "test|OPENUSDT",
                    "symbol": "OPENUSDT",
                    "state": "EARLY_PULSE",
                    "price": 1.0,
                },
                now=now,
            )
            symbols, intervals = build_order_book_collection_plan(
                data_dir,
                [],
                [],
                now=now.replace(hour=15),
                max_symbols=32,
                base_sample_seconds=1.0,
            )
            self.assertEqual(symbols, ["OPENUSDT"])
            self.assertEqual(intervals["OPENUSDT"], 15.0)

    def test_order_book_plan_prioritizes_new_signals_when_capped(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            now = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)
            (data_dir / "pulse_candidates.json").write_text(
                '{"NEWUSDT":{"detected_at":"2026-08-17T11:55:00+00:00"},'
                '"OLDUSDT":{"detected_at":"2026-08-17T08:00:00+00:00"}}'
            )
            symbols, intervals = build_order_book_collection_plan(
                data_dir,
                [],
                ["OLDUSDT", "NEWUSDT"],
                now=now,
                max_symbols=1,
                base_sample_seconds=1.0,
            )
            self.assertEqual(symbols, ["NEWUSDT"])
            self.assertEqual(intervals, {"NEWUSDT": 1.0})


if __name__ == "__main__":
    unittest.main()
