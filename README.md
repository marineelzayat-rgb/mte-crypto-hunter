# MTE Crypto Hunter

See [REPORT.md](REPORT.md) for the frozen 2022/2025/2026 Hunter→MTE backtest.
The current result validates candidate discovery, but rejects automated entry
after the untouched 2022 holdout; order execution is intentionally disabled.
The live service uses the frozen `MTE_PRE_HUNTER_RF_V0_3` model at its validated
`0.6753` threshold. It records order-book/taker-flow data for forward research;
it does not use that data as a proven entry filter yet.

An auditable, causal research and scanning pipeline for finding Binance USDT altcoins in the transition from dormant/base-building behavior into structural expansion.

The project deliberately separates:

1. **Research labels** — what counts as an explosion, defined without using future data in features.
2. **Pre-explosion ranking** — volume wake-up, compression, structure, and relative strength.
3. **MTE ignition** — a confirmed expansion break with no mandatory retest.
4. **Execution state** — `WATCH`, `ARMED`, `HUNTER_ALERT`, or `NO_CHASE`.
5. **Crypto flow research** — activation-time taker flow plus forward-recorded
   order-book/tape data. This is not enabled as an entry filter until it passes
   out-of-sample validation.

The default scanner is long-only and uses public Binance Spot market data. No API key is required. Futures open interest, funding, and taker-flow are deliberately an optional confirmation layer rather than a dependency.

## Quick start

```bash
cd mte_crypto_hunter
python -m mte_crypto.scan --top 120 --output output/latest_scan.csv
```

The scanner first filters all Binance USDT pairs by 24-hour quote volume, then downloads one-hour candles for the liquid universe, resamples causally to H4/D1, and ranks candidates.

To inspect one symbol:

```bash
python -m mte_crypto.scan --symbols ACEUSDT --output output/ace_scan.csv
```

Run the stateful monitor once (prints alerts when Telegram environment
variables are absent):

```bash
python -m mte_crypto.monitor --once
```

Continuous five-minute monitoring:

```bash
python -m mte_crypto.monitor --every-minutes 5
```

Optional Telegram delivery uses `MTE_TELEGRAM_BOT_TOKEN` and
`MTE_TELEGRAM_CHAT_ID`. Secrets are read from environment variables and are
never stored in the project.

Run tests:

```bash
python -m unittest discover -s tests -v
```

Attach causal flow fields to a completed Hunter→MTE result:

```bash
python -m mte_crypto.enrich_activation_flow \
  --input output/pre_hunter_mte_pivot_2026.csv \
  --history-dirs data/archive_2026 \
  --output output/pre_hunter_mte_pivot_2026_flow.csv
```

Record live top-20 depth plus executed taker flow for forward validation:

```bash
python -m mte_crypto.book_collector ACEUSDT SOLUSDT \
  --output output/live_order_book.jsonl
```

The collector is read-only and never places an order. A single visible wall is
not treated as evidence; persistence and replenishment must be measured across
the recorded snapshots.

## Railway 24/7 service

The included `Dockerfile` starts `python -m mte_crypto.daemon`. It runs two
causal discovery layers:

1. every hour, scans the top 120 liquid Binance Spot USDT altcoins and applies
   the exact causal `PRE_IGNITION_HUNT` gate plus frozen RF ranker;
2. every five minutes, takes a lightweight price/volume pulse of every eligible
   Spot/USDT market, including cold markets below the top-120 liquidity cutoff;
3. labels a pulse `EARLY_PULSE` while its 24h return is at most 20%, or
   `RAPID_MOVE_NO_CHASE` once it is already extended;
4. keeps each `HUNTER_ALERT` active for 48 hours and each pulse candidate active
   for two hours;
5. records top-20 order-book and executed taker-flow snapshots once per second
   for both hunter and pulse candidates;
6. writes scans, pulse history, and compressed JSONL research data under `/data`.
7. exposes a read-only live ledger at `/status` and `/status.json` after a
   Railway public domain is generated for the service.

Recommended Railway volume mount: `/data`. Optional environment variables:

- `MTE_SCAN_TOP` (default `120`)
- `MTE_SCAN_INTERVAL_SECONDS` (default `3600`)
- `MTE_PULSE_INTERVAL_SECONDS` (default `300`)
- `MTE_PULSE_TTL_MINUTES` (default `120`)
- `MTE_CANDIDATE_TTL_HOURS` (default `48`)
- `MTE_BOOK_SAMPLE_SECONDS` (default `1`)
- `MTE_MAX_BOOK_SYMBOLS` (default `10`)
- `MTE_TELEGRAM_BOT_TOKEN` and `MTE_TELEGRAM_CHAT_ID` for alerts

No Binance API key is used or needed. There is no order-placement code.
The public status endpoints use a strict market-field allowlist; Telegram
credentials, environment variables, and raw order-book files are never exposed.

Download historical monthly archives (includes delisted pairs when
`--all-historical` is used):

```bash
python -m mte_crypto.archive --start 2025-01 --end 2026-07 --all-historical --output-dir data/archive
python -m mte_crypto.research --data-dir data/archive --btc data/archive/BTCUSDT.csv
```

## Default research definition

An explosion is not a signal. It is an outcome label:

- forward 24-hour maximum return >= 40%, or
- forward 48-hour maximum return >= 60%.

Signals and features are computed only from information available at the timestamp being evaluated. The event-study layer compares explosion observations with time-matched non-explosion controls.

## Important limitations

- A current-symbol scan has survivorship bias. Full validation must include historical listings and delistings from the Binance public archive.
- A price/volume scanner cannot anticipate a completely untelegraphed listing or news event. Its job is to detect the first causal footprint before the terminal daily candle.
- `RESEARCH_IGNITION` is the frozen MTE Crypto v0.1 candidate: H1 break plus
  expansion quality, 5–10% prior 24h return, 5–10% structural risk, RVOL
  2–10x, 24h-volume ratio 1.5–10x, at least $20m rolling spot volume,
  H4 compression rank <= 0.20, green-volume share >= 60%, positive relative
  strength, and BTC 24h return >= -2%. It does **not** require a retest.
- `ARMED` is an early research watch, not an entry instruction.
- The project is paper/alert automation only; it never places an exchange
  order or stores credentials.
- Real execution must include spread, slippage, fees, funding, and exchange-specific wick behavior.
