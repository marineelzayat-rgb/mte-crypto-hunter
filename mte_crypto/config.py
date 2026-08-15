from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerConfig:
    api_base: str = "https://data-api.binance.vision"
    interval: str = "1h"
    kline_limit: int = 1000
    min_quote_volume_24h: float = 1_000_000.0
    max_universe: int = 500
    workers: int = 20

    atr_length: int = 14
    pivot_strength: int = 3
    breakout_lookback_h: int = 48
    macro_high_lookback_h: int = 480
    wakeup_lookback_h: int = 168

    min_expansion_tr_atr: float = 1.05
    min_expansion_body_ratio: float = 0.55
    min_expansion_clv: float = 0.68
    min_ignition_rvol: float = 2.0
    break_buffer_atr: float = 0.10
    no_chase_atr: float = 1.50
    no_chase_return_24h: float = 0.35

    watch_score: float = 50.0
    armed_score: float = 65.0


DEFAULT_CONFIG = ScannerConfig()
