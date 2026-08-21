from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from mte_crypto.live_futures import (
    BinanceApiError,
    BinanceFuturesPrivateClient,
    LIVE_CONFIRMATION,
    LiveFuturesConfig,
    close_live_futures_positions,
    live_futures_payload,
    open_live_futures_position,
    refresh_futures_connection_status,
)


UTC = timezone.utc


class FakeClient:
    def __init__(
        self,
        *,
        can_trade=True,
        hedge_mode=False,
        multi_assets=False,
        fail_stop=False,
        fail_flatten=False,
    ):
        self.can_trade = can_trade
        self.hedge_mode = hedge_mode
        self.multi_assets = multi_assets
        self.fail_stop = fail_stop
        self.fail_flatten = fail_flatten
        self.orders = []
        self.algo_orders = []
        self.margin_changes = []
        self.leverage_changes = []
        self.position_open = True
        self.algo_status = {"algoStatus": "NEW"}

    def sync_time(self):
        return 0

    def account_information(self):
        return {
            "canTrade": self.can_trade,
            "canWithdraw": False,
            "canDeposit": True,
        }

    def account_config(self):
        return {
            "dualSidePosition": self.hedge_mode,
            "multiAssetsMargin": self.multi_assets,
        }

    def balances(self):
        return [
            {"asset": "USDT", "balance": "100.00", "availableBalance": "100.00"}
        ]

    def exchange_info(self, symbol):
        return {
            "symbols": [
                {
                    "symbol": "AAAUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "filters": [],
                },
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "filters": [
                        {
                            "filterType": "MARKET_LOT_SIZE",
                            "stepSize": "0.001",
                            "minQty": "0.001",
                            "maxQty": "100000",
                        },
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }

    def positions(self, symbol=None):
        amount = "2.2" if self.position_open else "0"
        return [{"symbol": symbol, "positionSide": "BOTH", "positionAmt": amount}]

    def change_margin_type(self, symbol, margin_type):
        self.margin_changes.append((symbol, margin_type))
        return {"code": 200}

    def change_leverage(self, symbol, leverage):
        self.leverage_changes.append((symbol, leverage))
        return {"leverage": leverage}

    def new_order(self, **params):
        self.orders.append(params)
        if params["side"] == "BUY":
            return {
                "orderId": 10,
                "executedQty": "2.2",
                "avgPrice": "10.00",
                "cumQuote": "22.00",
            }
        if self.fail_flatten:
            raise BinanceApiError("flatten rejected", status=400, code=-1)
        self.position_open = False
        return {
            "orderId": 12,
            "executedQty": params["quantity"],
            "avgPrice": "11.00",
        }

    def query_order(self, symbol, order_id=None, client_order_id=None):
        return {"orderId": order_id or 10, "executedQty": "2.2", "avgPrice": "10"}

    def new_algo_order(self, **params):
        if self.fail_stop:
            raise RuntimeError("stop rejected")
        self.algo_orders.append(params)
        return {"algoId": 20, "algoStatus": "NEW"}

    def query_algo_order(self, algo_id=None, client_algo_id=None):
        if self.fail_stop:
            raise RuntimeError("stop absent")
        return {"algoId": algo_id or 20, **self.algo_status}

    def cancel_algo_order(self, algo_id=None, client_algo_id=None):
        return {"algoId": algo_id or 20, "algoStatus": "CANCELED"}


class LiveFuturesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.now = datetime(2026, 8, 21, 18, tzinfo=UTC)
        self.snapshot = {"symbol": "TESTUSDT", "bid": 9.99, "ask": 10.0}

    def tearDown(self):
        self.temporary.cleanup()

    def _armed(self):
        return LiveFuturesConfig(enabled=True, confirmation=LIVE_CONFIRMATION)

    def _paper_open(self):
        return {
            "opened": True,
            "symbol": "TESTUSDT",
            "slot_id": 1,
            "alert_id": "alert-1",
        }

    def test_rsa_client_loads_generated_key_and_rejects_ed25519(self):
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        client = BinanceFuturesPrivateClient("api", pem)
        self.assertEqual(client.api_key, "api")

        wrong = ed25519.Ed25519PrivateKey.generate().private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        with self.assertRaises(ValueError):
            BinanceFuturesPrivateClient("api", wrong)

    def test_connection_status_is_read_only_and_redacts_credentials(self):
        with patch.dict(
            "os.environ",
            {
                "BINANCE_FUTURES_API_KEY": "secret-api",
                "BINANCE_FUTURES_RSA_PRIVATE_KEY": "private-secret-value",
            },
            clear=False,
        ):
            status = refresh_futures_connection_status(
                self.data_dir,
                now=self.now,
                cfg=LiveFuturesConfig(),
                client=FakeClient(can_trade=False),
            )
        self.assertTrue(status["connected"])
        self.assertFalse(status["can_trade"])
        stored = (self.data_dir / "live_futures_connection.json").read_text()
        self.assertNotIn("secret-api", stored)
        self.assertNotIn("private-secret-value", stored)

    def test_entry_requires_double_confirmation(self):
        result = open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=LiveFuturesConfig(enabled=True, confirmation="wrong"),
            client=FakeClient(),
        )
        self.assertEqual(result["reason"], "SAFE_DISABLED")

    def test_armed_entry_uses_isolated_4x_compounding_and_exchange_algo_stop(self):
        client = FakeClient()
        result = open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=self._armed(),
            client=client,
        )
        self.assertTrue(result["opened"])
        self.assertEqual(client.margin_changes, [("TESTUSDT", "ISOLATED")])
        self.assertEqual(client.leverage_changes, [("TESTUSDT", 4)])
        self.assertEqual(client.orders[0]["side"], "BUY")
        self.assertEqual(client.orders[0]["quantity"], "4.4")
        self.assertEqual(result["entry_margin"], 11.0)
        self.assertEqual(result["wallet_balance_at_entry"], 100.0)
        self.assertEqual(result["daily_loss_limit_at_entry"], 30.0)
        self.assertEqual(client.algo_orders[0]["type"], "STOP_MARKET")
        self.assertEqual(client.algo_orders[0]["triggerPrice"], "9.25")
        self.assertEqual(client.algo_orders[0]["workingType"], "MARK_PRICE")
        self.assertEqual(client.algo_orders[0]["reduceOnly"], "true")

    def test_compounding_margin_tracks_realized_wallet_balance(self):
        class GrownClient(FakeClient):
            def balances(self):
                return [
                    {
                        "asset": "USDT",
                        "balance": "150.00",
                        "availableBalance": "150.00",
                    }
                ]

            def new_order(self, **params):
                self.orders.append(params)
                if params["side"] == "BUY":
                    return {
                        "orderId": 10,
                        "executedQty": params["quantity"],
                        "avgPrice": "10.00",
                        "cumQuote": "66.00",
                    }
                return super().new_order(**params)

        client = GrownClient()
        result = open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=self._armed(),
            client=client,
        )
        self.assertTrue(result["opened"])
        self.assertEqual(result["entry_margin"], 16.5)
        self.assertEqual(result["daily_loss_limit_at_entry"], 45.0)
        self.assertEqual(client.orders[0]["quantity"], "6.6")

    def test_equity_floor_blocks_new_entries(self):
        class DepletedClient(FakeClient):
            def balances(self):
                return [
                    {
                        "asset": "USDT",
                        "balance": "10.00",
                        "availableBalance": "10.00",
                    }
                ]

        client = DepletedClient()
        result = open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=self._armed(),
            client=client,
        )
        self.assertEqual(result["reason"], "EXPERIMENT_EQUITY_FLOOR")
        self.assertEqual(client.orders, [])

    def test_daily_loss_gate_blocks_after_thirty_percent_realized_loss(self):
        state = {
            "version": 1,
            "positions": {},
            "closed_trades": [
                {"closed_at": self.now.isoformat(), "pnl": -30.0}
            ],
            "events": [],
        }
        (self.data_dir / "live_futures.json").write_text(json.dumps(state))
        client = FakeClient()
        result = open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=self._armed(),
            client=client,
        )
        self.assertEqual(result["reason"], "DAILY_LOSS_KILL_SWITCH")
        self.assertEqual(client.orders, [])

    def test_hedge_mode_is_rejected_before_any_order(self):
        client = FakeClient(hedge_mode=True)
        result = open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=self._armed(),
            client=client,
        )
        self.assertEqual(result["reason"], "HEDGE_MODE_UNSUPPORTED")
        self.assertEqual(client.orders, [])

    def test_stop_failure_immediately_flattens_reduce_only(self):
        client = FakeClient(fail_stop=True)
        with self.assertRaises(RuntimeError):
            open_live_futures_position(
                self.data_dir,
                self._paper_open(),
                self.snapshot,
                now=self.now,
                cfg=self._armed(),
                client=client,
            )
        self.assertEqual(len(client.orders), 2)
        self.assertEqual(client.orders[1]["side"], "SELL")
        self.assertEqual(client.orders[1]["reduceOnly"], "true")

    def test_unknown_emergency_flatten_is_durable_and_retried_next_cycle(self):
        client = FakeClient(fail_stop=True, fail_flatten=True)
        with self.assertRaises(BinanceApiError):
            open_live_futures_position(
                self.data_dir,
                self._paper_open(),
                self.snapshot,
                now=self.now,
                cfg=self._armed(),
                client=client,
            )
        self.assertEqual(live_futures_payload(self.data_dir)["open_count"], 1)

        client.fail_flatten = False
        closed = close_live_futures_positions(
            self.data_dir, [], now=self.now, cfg=self._armed(), client=client
        )
        self.assertEqual(closed[0]["exit_reason"], "EMERGENCY_UNPROTECTED_FLATTEN")
        self.assertEqual(live_futures_payload(self.data_dir)["open_count"], 0)

    def test_exchange_stop_is_reconciled(self):
        client = FakeClient()
        cfg = self._armed()
        open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=cfg,
            client=client,
        )
        client.position_open = False
        client.algo_status = {
            "algoStatus": "FINISHED",
            "actualOrderId": 30,
            "actualQty": "2.2",
            "actualPrice": "9.25",
        }
        closed = close_live_futures_positions(
            self.data_dir, [], now=self.now, cfg=cfg, client=client
        )
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "EXCHANGE_HARD_STOP")
        self.assertEqual(live_futures_payload(self.data_dir)["open_count"], 0)

    def test_paper_exit_closes_reduce_only_and_public_payload_is_redacted(self):
        client = FakeClient()
        cfg = self._armed()
        open_live_futures_position(
            self.data_dir,
            self._paper_open(),
            self.snapshot,
            now=self.now,
            cfg=cfg,
            client=client,
        )
        payload = live_futures_payload(self.data_dir)
        serialized = str(payload)
        self.assertNotIn("TESTUSDT", serialized)
        self.assertNotIn("entry_order_id", serialized)
        self.assertNotIn("usdt_available", serialized)

        closed = close_live_futures_positions(
            self.data_dir,
            [{"symbol": "TESTUSDT", "exit_reason": "TRAIL_STOP"}],
            now=self.now,
            cfg=cfg,
            client=client,
        )
        self.assertEqual(len(closed), 1)
        self.assertEqual(client.orders[-1]["side"], "SELL")
        self.assertEqual(client.orders[-1]["reduceOnly"], "true")


if __name__ == "__main__":
    unittest.main()
