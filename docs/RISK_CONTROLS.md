# Risk Controls

## Philosophy

This system is designed with the assumption that **the default behavior should be to NOT trade**. Every order must pass through multiple independent safety checks. A single failed check blocks the order.

The system does not promise profits. It is designed to protect capital.

## Pre-Trade Risk Checks

All checks execute sequentially. ALL must pass for an order to be submitted.

### 1. Emergency Stop File
- **What**: Checks for a file named `EMERGENCY_STOP` in the project root.
- **How to trigger**: `touch EMERGENCY_STOP`
- **How to clear**: `rm EMERGENCY_STOP`
- **Effect**: Blocks all orders immediately.

### 2. Circuit Breaker
- **What**: A software kill switch that requires manual reset.
- **Triggers**: Max daily loss exceeded, max consecutive losses.
- **Reset**: Requires restarting the bot or calling `risk_manager.reset_circuit_breaker()`.

### 3. Daily Loss Limit
- **Default**: $10 (configurable via `MAX_DAILY_LOSS`)
- **Calculation**: Realized PnL + unrealized PnL since start of day.
- **Effect**: Trips the circuit breaker.

### 4. Consecutive Losses
- **Default**: 5 (configurable via `MAX_CONSECUTIVE_LOSSES`)
- **Effect**: Trips the circuit breaker.

### 5. Order Frequency
- **Default**: 6 orders per minute (configurable via `MAX_ORDERS_PER_MINUTE`)
- **Effect**: Temporarily blocks new orders until rate drops.

### 6. Stale Data Lockout
- **Threshold**: 30 seconds since last WebSocket update.
- **Effect**: Blocks all orders until fresh data arrives.

### 7. Spread Guard
- **Min spread**: 1 cent (`MIN_SPREAD_THRESHOLD`) — avoids trading in extremely tight markets.
- **Max spread**: 15 cents (`MAX_SPREAD_THRESHOLD`) — avoids illiquid markets with wide spreads.

### 8. Liquidity Depth Guard
- **Default**: $20 combined depth within 5 cents (`MIN_LIQUIDITY_DEPTH`)
- **Effect**: Blocks orders in thin books.

### 9. Slippage Guard
- **Default**: 3 cents max deviation from midpoint (`MAX_SLIPPAGE`)
- **Effect**: Prevents orders priced far from fair value.

### 10. Market Exposure Limit
- **Default**: $10 per market (`MAX_POSITION_PER_MARKET`)
- **Effect**: Caps concentration in any single market.

### 11. Total Exposure Limit
- **Default**: $50 across all markets (`MAX_TOTAL_EXPOSURE`)
- **Effect**: Caps overall portfolio risk.

### 12. Volatility Lockout
- **Threshold**: 10% 1-minute volatility.
- **Effect**: Blocks orders during abnormally volatile periods.

### 13. Cash Sufficiency
- **What**: Verifies available cash covers the order cost before submission.
- **Effect**: Rejects buy orders that would exceed available cash.

## Three Safety Gates for Live Trading

Live trading requires **all three** of these to be explicitly set:

| Gate | Environment Variable | Default |
|------|---------------------|---------|
| 1 | `DRY_RUN=false` | true |
| 2 | `ENABLE_LIVE_TRADING=true` | false |
| 3 | `LIVE_TRADING_ACKNOWLEDGED=true` | false |

Plus valid API credentials (`PRIVATE_KEY`, `POLY_API_KEY`, `POLY_API_SECRET`).

If any gate is missing, the system refuses to submit real orders. There are also code-level safety re-checks in the trading client that block live orders even if configuration is somehow bypassed.

## AI Cannot Override Risk

The decision engine (three-layer ensemble) produces trade candidates, but **risk management is always deterministic**. No AI signal — regardless of how strongly all three intelligence layers agree — can bypass or weaken any risk check. The risk manager sits between the decision engine and execution, and it has absolute veto power.

## Additional Safeguards

### Order Deduplication
The execution engine rejects orders with the same token + side + price as an existing active order.

### Stale Order Cleanup
Orders older than 5 minutes are automatically canceled by the housekeeping loop.

### Limit Orders Only
Market orders are not supported. All orders are limit orders to prevent unexpected fills at bad prices.

### Manual Kill Switch
At any time, create an `EMERGENCY_STOP` file to halt all trading:
```bash
touch EMERGENCY_STOP    # halt
rm EMERGENCY_STOP       # resume
```

## Recommended Starting Parameters

For a $100 bankroll, first-time user:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| DRY_RUN | true | Always start in simulation |
| DEFAULT_ORDER_SIZE | $1.00 | Tiny size for learning |
| MAX_POSITION_PER_MARKET | $10.00 | 10% of bankroll per market |
| MAX_TOTAL_EXPOSURE | $50.00 | 50% max deployment |
| MAX_DAILY_LOSS | $10.00 | 10% daily drawdown limit |
| MAX_ORDERS_PER_MINUTE | 6 | Prevent overtrading |
| MAX_CONSECUTIVE_LOSSES | 5 | Auto-halt on losing streak |

## Monitoring Risk in Production

1. Watch structured logs for `risk_check_failed` and `CIRCUIT_BREAKER_TRIPPED` events.
2. Periodically run `python scripts/export_pnl_report.py` to review performance.
3. Monitor the `metrics.snapshot()` counters for `risk_rejections` and `orders_rejected_risk`.
4. Keep the emergency stop file mechanism ready.
