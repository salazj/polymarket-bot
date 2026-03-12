"""Stock feature engine — computes StockFeatures from bars and quotes."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

from app.monitoring import get_logger
from app.stocks.models import StockBar, StockFeatures
from app.utils.helpers import utc_now

logger = get_logger(__name__)


class StockFeatureEngine:
    """Maintains rolling windows of bars and computes technical features."""

    def __init__(self, symbol: str, max_bars: int = 200) -> None:
        self._symbol = symbol
        self._bars: deque[StockBar] = deque(maxlen=max_bars)
        self._last_quote: dict[str, float] = {}
        self._high_of_day: float = 0.0
        self._low_of_day: float = float("inf")
        self._day_volume: int = 0
        self._day_start: datetime | None = None

    def add_bar(self, bar: StockBar) -> None:
        self._bars.append(bar)
        if bar.high > self._high_of_day:
            self._high_of_day = bar.high
        if bar.low < self._low_of_day:
            self._low_of_day = bar.low
        self._day_volume += bar.volume

    def update_quote(self, bid: float, ask: float, last: float) -> None:
        self._last_quote = {"bid": bid, "ask": ask, "last": last}

    def start_new_day(self) -> None:
        self._high_of_day = 0.0
        self._low_of_day = float("inf")
        self._day_volume = 0
        self._day_start = utc_now()

    def compute(self) -> StockFeatures:
        now = utc_now()
        closes = [b.close for b in self._bars]
        highs = [b.high for b in self._bars]
        lows = [b.low for b in self._bars]
        volumes = [b.volume for b in self._bars]

        last = self._last_quote.get("last", closes[-1] if closes else 0.0)
        bid = self._last_quote.get("bid", 0.0)
        ask = self._last_quote.get("ask", 0.0)

        return StockFeatures(
            symbol=self._symbol,
            timestamp=now,
            last_price=last,
            bid=bid,
            ask=ask,
            spread=ask - bid if bid > 0 and ask > 0 else 0.0,
            volume_1m=volumes[-1] if volumes else 0,
            volume_5m=sum(volumes[-5:]) if len(volumes) >= 5 else sum(volumes),
            volume_today=self._day_volume,
            vwap=self._compute_vwap(),
            rsi_14=self._compute_rsi(closes, 14),
            sma_20=self._sma(closes, 20),
            ema_9=self._ema(closes, 9),
            atr_14=self._compute_atr(highs, lows, closes, 14),
            volatility_1h=self._compute_volatility(closes, 60),
            momentum_1m=self._momentum(closes, 1),
            momentum_5m=self._momentum(closes, 5),
            momentum_15m=self._momentum(closes, 15),
            price_vs_vwap=last - self._compute_vwap() if last > 0 else 0.0,
            high_of_day=self._high_of_day if self._high_of_day > 0 else last,
            low_of_day=self._low_of_day if self._low_of_day < float("inf") else last,
            relative_volume=1.0,
        )

    def _compute_vwap(self) -> float:
        if not self._bars:
            return 0.0
        total_pv = sum(b.close * b.volume for b in self._bars)
        total_v = sum(b.volume for b in self._bars)
        return total_pv / total_v if total_v > 0 else 0.0

    @staticmethod
    def _compute_rsi(closes: list[float], period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-period:]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _sma(values: list[float], period: int) -> float:
        if len(values) < period:
            return values[-1] if values else 0.0
        return sum(values[-period:]) / period

    @staticmethod
    def _ema(values: list[float], period: int) -> float:
        if not values:
            return 0.0
        k = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * k + ema * (1 - k)
        return ema

    @staticmethod
    def _compute_atr(
        highs: list[float], lows: list[float], closes: list[float], period: int
    ) -> float:
        if len(closes) < 2:
            return 0.0
        trs: list[float] = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0
        return sum(trs[-period:]) / period

    @staticmethod
    def _compute_volatility(closes: list[float], window: int) -> float:
        if len(closes) < 2:
            return 0.0
        recent = closes[-window:]
        if len(recent) < 2:
            return 0.0
        mean = sum(recent) / len(recent)
        variance = sum((c - mean) ** 2 for c in recent) / len(recent)
        return variance**0.5

    @staticmethod
    def _momentum(closes: list[float], lookback: int) -> float:
        if len(closes) <= lookback:
            return 0.0
        prev = closes[-lookback - 1]
        if prev == 0:
            return 0.0
        return (closes[-1] - prev) / prev
