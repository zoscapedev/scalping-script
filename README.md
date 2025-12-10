# BTC Live Scalper

Live trading bot for BTC scalping using Hyperliquid exchange.

## Deploy to Railway

1. Install Railway CLI:
```bash
npm i -g @railway/cli
```

2. Login to Railway:
```bash
railway login
```

3. Initialize and deploy:
```bash
cd /media/omchillure/Projects/scalping/scalping
railway init
railway up
```

## Configuration

The bot uses the following parameters:
- Symbol: BTC
- Timeframe: 5m
- Leverage: 40x
- Initial Capital: $200

## Features

- Follow Line indicator with Bollinger Bands
- ATR-based stop loss and take profit
- SR (Support/Resistance) levels
- Real-time candle updates via WebSocket
