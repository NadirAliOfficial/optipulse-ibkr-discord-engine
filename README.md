# IBKR Options Chain Scanner - Setup Guide

## 📋 What You Need

1. **Interactive Brokers Account** (Paper or Live)
2. **IB Gateway** installed and running
3. **Python 3.8+** installed
4. **Discord Webhook** URL (for alerts)

---

## 🚀 Quick Start (5 Steps)

### Step 1: Install Python Dependencies

Open your terminal/command prompt and run:

```bash
pip install ib_insync requests
```

**What this does:** Installs the libraries needed to connect to IBKR and send Discord messages.

---

### Step 2: Setup IB Gateway

1. **Download IB Gateway** from Interactive Brokers website
2. **Login** to IB Gateway with your credentials
3. **Important Settings:**
   - Go to **Configure → Settings → API → Settings**
   - ✅ Enable "Enable ActiveX and Socket Clients"
   - ✅ Check "Read-Only API"
   - Set **Socket Port:**
     - **4002** for Paper Trading (recommended for testing)
     - **4001** for Live Trading
   - ✅ Check "Allow connections from localhost only"
   - Click **OK** and restart IB Gateway

**Screenshot reference:** The settings should look like this:
```
[✓] Enable ActiveX and Socket Clients
[✓] Read-Only API
Socket port: 4002
[✓] Allow connections from localhost only
```

---

### Step 3: Create Discord Webhook

1. **Open Discord** and go to your server
2. **Right-click** the channel where you want alerts
3. **Click** "Edit Channel" → "Integrations" → "Webhooks"
4. **Click** "New Webhook"
5. **Name it** (e.g., "IBKR Scanner")
6. **Copy** the Webhook URL (looks like: `https://discord.com/api/webhooks/...`)

---

### Step 4: Configure the Script

1. **Open** `phase1_chain_scanner.py` in any text editor
2. **Find** the configuration section at the top (around line 20)
3. **Edit** these values:

```python
# REQUIRED: Add your Discord webhook URL
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE'

# REQUIRED: Set correct IB Gateway port
IB_PORT = 4002  # Use 4002 for paper trading, 4001 for live

# OPTIONAL: Adjust filters if needed
DELTA_MIN = 0.35           # Keep options with delta between 0.35-0.55
DELTA_MAX = 0.55
MAX_SPREAD_PERCENT = 7.0   # Reject if bid-ask spread > 7%
MIN_VOLUME = 10            # Minimum daily volume
MIN_OPEN_INTEREST = 50     # Minimum open interest
SCAN_INTERVAL_SECONDS = 60 # Scan every 60 seconds
```

4. **Save** the file

---

### Step 5: Run the Scanner

1. **Make sure IB Gateway is running** and logged in
2. **Open terminal** in the folder containing `phase1_chain_scanner.py`
3. **Run:**

```bash
python phase1_chain_scanner.py
```

**You should see:**
```
🚀 Starting IBKR Options Scanner
📊 Symbol: SHOP
🎯 Delta range: 0.35 - 0.55
💰 Max spread: 7.0%
📈 Min volume: 10, Min OI: 50
⏱️  Scan interval: 60 seconds
======================================================================
🔌 Connecting to IB Gateway at 127.0.0.1:4002...
✅ Connected to IBKR successfully!
🔍 Starting scan for SHOP...
```

---

## 📊 What the Scanner Does

### Real-Time Process:

1. **Connects** to your IB Gateway (read-only, safe)
2. **Fetches** all SHOP option contracts (calls and puts)
3. **Gets** live market data:
   - Bid/Ask prices
   - Delta values
   - Volume & Open Interest
4. **Filters** options using your criteria:
   - ✅ Delta between 0.35-0.55
   - ✅ Spread < 7%
   - ✅ Volume ≥ 10
   - ✅ Open Interest ≥ 50
5. **Sends** qualifying options to Discord
6. **Repeats** every 60 seconds

---

## 🔧 Configuration Explained

### Delta Filter
```python
DELTA_MIN = 0.35
DELTA_MAX = 0.55
```
- **Delta** measures how much option price moves with $1 stock move
- **0.35-0.55** is the "sweet spot" for balanced risk/reward
- Lower delta (0.35) = cheaper, further OTM
- Higher delta (0.55) = more expensive, closer to ATM

### Spread Filter
```python
MAX_SPREAD_PERCENT = 7.0
```
- **Spread** = difference between bid and ask price
- **7%** means you reject options with wide spreads
- Tight spreads = easier to enter/exit positions
- If spread > 7%, the option is rejected (not liquid enough)

### Liquidity Filters
```python
MIN_VOLUME = 10
MIN_OPEN_INTEREST = 50
```
- **Volume** = contracts traded today
- **Open Interest** = total open contracts
- Higher values = more liquid (easier to buy/sell)

### Scan Interval
```python
SCAN_INTERVAL_SECONDS = 60
```
- How often the script scans for new options
- **60 seconds** = once per minute (balanced)
- Lower = more frequent updates (more API calls)
- Higher = less frequent (lighter on resources)

---

## 📱 Discord Alert Format

When an option passes all filters, you'll get:

```
🎯 New Option Alert

Stock: SHOP
Type: CALL
Expiry: 20250221
Strike: $85.00

Pricing:
• Bid: $2.45
• Ask: $2.60
• Spread: 5.77%

Greeks & Volume:
• Delta: 0.452
• Volume: 125
• Open Interest: 450

Scanned at 2026-02-01 14:30:15
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Failed to connect to IBKR"

**Solutions:**
1. ✅ IB Gateway is running and logged in
2. ✅ Port number matches (4002 for paper, 4001 for live)
3. ✅ "Enable ActiveX and Socket Clients" is checked in IB settings
4. ✅ "Read-Only API" is enabled
5. Try restarting IB Gateway

### Issue 2: "No option chains found for SHOP"

**Solutions:**
1. Market might be closed (options trade 9:30 AM - 4:00 PM ET)
2. Check if SHOP symbol is correct
3. IB Gateway might not have data permissions for options
4. Try with a different symbol (e.g., SPY, AAPL)

### Issue 3: "Discord webhook failed"

**Solutions:**
1. ✅ Webhook URL is correct (copied fully)
2. ✅ Discord channel still exists
3. ✅ Webhook wasn't deleted in Discord settings
4. Test webhook manually: Visit https://discord.com/developers/docs/resources/webhook

### Issue 4: Script crashes or stops

**Solutions:**
1. Check IB Gateway didn't log out automatically
2. Look at error message in terminal
3. Run with debug mode: Change `LOG_LEVEL = logging.DEBUG` in config
4. The script has auto-reconnect, but if Gateway restarts you may need to restart script

### Issue 5: No alerts being sent

**Possible reasons:**
1. No options currently meet all filter criteria (delta + spread + volume + OI)
2. Options were already sent (duplicate detection working)
3. Market is closed or low activity period
4. Try loosening filters temporarily to test:
   ```python
   DELTA_MIN = 0.20
   DELTA_MAX = 0.80
   MAX_SPREAD_PERCENT = 15.0
   MIN_VOLUME = 5
   MIN_OPEN_INTEREST = 10
   ```

---

## 🛑 How to Stop the Scanner

Press **Ctrl+C** (or Cmd+C on Mac) in the terminal.

You'll see:
```
⚠️  Shutdown requested by user
🔌 Disconnected from IBKR
👋 Scanner stopped
```

---

## 📈 Performance Tips

### For Better Results:

1. **Run during market hours** (9:30 AM - 4:00 PM ET)
   - Options have the most activity during this time
   - More likely to find options meeting liquidity filters

2. **Start with looser filters** if testing
   - Lower MIN_VOLUME and MIN_OPEN_INTEREST
   - Increase MAX_SPREAD_PERCENT
   - Once you see alerts flowing, tighten them

3. **Adjust scan interval based on needs**
   - **30 seconds** = more real-time (higher load)
   - **60 seconds** = balanced (recommended)
   - **120 seconds** = lighter, slower updates

4. **Monitor the logs**
   - Green ✅ = success
   - Yellow ⚠️ = warning (non-critical)
   - Red ❌ = error (needs attention)

---

## 🔒 Safety Features Built-In

✅ **Read-only connection** - Cannot place trades
✅ **Auto-reconnect** - Recovers from connection drops  
✅ **Error handling** - Won't crash on bad data
✅ **Duplicate prevention** - Won't spam same option
✅ **Rate limiting** - Won't overwhelm Discord
✅ **Market data validation** - Skips incomplete data

---

## 📝 What This Script Does NOT Do

❌ No trading or order placement
❌ No AI or machine learning
❌ No trading signals or recommendations
❌ No position management
❌ No P&L tracking

**This is purely a data scanner and filter.**

---

## 🎯 Next Steps (After You're Comfortable)

1. **Try different symbols** - Edit `SYMBOL = 'SHOP'` to 'SPY', 'AAPL', etc.
2. **Adjust filters** - Tune delta range, spread limits based on what you see
3. **Multiple instances** - Run multiple scripts for different symbols
4. **Save to file** - Modify to log results to CSV for analysis

---

## 🆘 Need Help?

If something isn't working:

1. **Check the logs** - The script prints detailed info
2. **Enable debug mode** - Set `LOG_LEVEL = logging.DEBUG`
3. **Test IB connection** - Make sure IB Gateway shows "Connected" status
4. **Verify Discord webhook** - Send a test message using an online tool

---

## 📚 Additional Resources

- **IB API Documentation**: https://interactivebrokers.github.io/tws-api/
- **ib_insync Documentation**: https://ib-insync.readthedocs.io/
- **Discord Webhooks Guide**: https://discord.com/developers/docs/resources/webhook
- **Interactive Brokers Gateway**: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

---

**Happy Scanning! 🚀**