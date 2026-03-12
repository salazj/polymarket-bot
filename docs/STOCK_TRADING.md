# Stock Trading Guide

$alazar-Trader supports stock (equities) trading via the Alpaca broker adapter.

## Setup

### 1. Get Alpaca API Keys

1. Create an account at [alpaca.markets](https://alpaca.markets)
2. Go to Dashboard > API Keys
3. Generate a new API key pair

### 2. Configure `.env`

```env
ASSET_CLASS=equities
BROKER=alpaca

ALPACA_API_KEY=your-key-id
ALPACA_SECRET_KEY=your-secret-key
ALPACA_PAPER=true    # Use paper trading (strongly recommended)
```

### 3. Stock Universe Selection

Two modes:

**Manual mode** — trade only specific tickers:
```env
STOCK_UNIVERSE_MODE=manual
STOCK_TICKERS=AAPL,MSFT,NVDA,TSLA,AMZN
```

**Filtered mode** — dynamically select stocks:
```env
STOCK_UNIVERSE_MODE=filtered
STOCK_MIN_VOLUME=100000
STOCK_MIN_PRICE=5.0
STOCK_MAX_PRICE=500.0
STOCK_SECTOR_INCLUDE=technology,healthcare
MAX_STOCK_SYMBOLS=20
```

### 4. Risk Limits

```env
STOCK_MAX_POSITION_DOLLARS=1000.0
STOCK_MAX_PORTFOLIO_DOLLARS=10000.0
STOCK_MAX_DAILY_LOSS_DOLLARS=500.0
STOCK_MAX_OPEN_POSITIONS=10
STOCK_MAX_ORDERS_PER_MINUTE=10
ALLOW_EXTENDED_HOURS=false
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `stock_momentum` | EMA-9 crossover + volume surge + RSI filter |
| `stock_mean_reversion` | Buy below VWAP with low RSI, sell at VWAP |
| `stock_breakout` | Buy on high-of-day breakout with volume confirmation |
| `stock_news_gated` | Gate entries on symbols with bullish NLP signals |

## How Stocks Differ from Prediction Markets

- **Market hours**: Stock trading respects NYSE hours (9:30-16:00 ET). Extended hours can be enabled.
- **Risk units**: Stock risk is dollar-based (max $1,000 per position) rather than contract-based.
- **Strategies**: Stock strategies use technical indicators (RSI, EMA, ATR, VWAP) rather than probability models.
- **Order types**: Stocks support market, limit, stop, and stop-limit orders.
- **Universe**: Stock universe selection uses volume, price, and sector filters.

## Dry Run Default

Stock trading defaults to `DRY_RUN=true`. In dry-run mode:
- No API calls are made to Alpaca
- Orders are simulated locally
- All strategies, risk checks, and features run normally
- Logs show `[DRY]` prefix for simulated fills

To enable paper trading (real API calls to Alpaca's paper environment):
```env
DRY_RUN=false
ENABLE_LIVE_TRADING=true
LIVE_TRADING_ACKNOWLEDGED=true
ALPACA_PAPER=true
```
