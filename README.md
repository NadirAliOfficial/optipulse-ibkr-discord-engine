# OptiPulse – IBKR Discord Options Engine

OptiPulse is a decision-support system for options traders using Interactive Brokers (IBKR).

It connects to IBKR via `ib_insync`, scans real-time options chains, evaluates multi-timeframe market conditions, and delivers structured alerts to Discord.

This project does **not** place trades. All execution is manual.

## Features
- Real-time IBKR options chain scanning
- Delta, spread, and liquidity filtering
- Multi-timeframe signal engine (1m / 5m / 15m)
- VWAP, EMA, RSI, ATR, and volume analysis
- Scored alerts with contract ranking
- Discord webhook alerts with chart snapshots

## Requirements
- Interactive Brokers account
- IB Gateway or TWS
- Python 3.10+
- Discord webhook


## Disclaimer
For educational and informational purposes only.
