import time, os, logging
from datetime import datetime
from dotenv import load_dotenv
from ib_insync import IB, Stock, Option
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Config
IB_HOST, IB_PORT, IB_CLIENT_ID = "127.0.0.1", 7497, 1
SYMBOL = "SHOP"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

DELTA_MIN, DELTA_MAX = 0.30, 0.60
MIN_VOL, MIN_OI = 0, 0  # Set to 0 for testing, increase later
STRIKE_RANGE = 0.10
MAX_EXPIRATIONS = 2
SCAN_INTERVAL = 300
BATCH_SIZE = 40
DEBUG = True  # Print all options with Greeks

sent_alerts = set()
no_data_alert_sent = False


def send_discord(msg):
    if not WEBHOOK_URL:
        logging.warning("⚠️  WEBHOOK_URL not set")
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=5)
    except Exception as e:
        logging.error(f"Discord error: {e}")


def send_alert(d):
    msg = (
        f"🎯 **{d['symbol']} {d['right']} Option Found!**\n"
        f"📅 Expiry: {d['expiry']}\n"
        f"💰 Strike: ${d['strike']}\n"
        f"💵 Price: ${d['price']:.2f}\n"
        f"📊 Delta: {d['delta']:.2f} | IV: {d['iv']:.1%}\n"
        f"📈 Volume: {d['vol']} | OI: {d['oi']}\n"
        f"⏰ {datetime.now().strftime('%I:%M:%S %p')}"
    )
    send_discord(msg)


def scan():
    global no_data_alert_sent

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=60, readonly=True)
    ib.reqMarketDataType(3)

    stock = Stock(SYMBOL, "SMART", "USD")
    ib.qualifyContracts(stock)

    [ticker] = ib.reqTickers(stock)
    ib.sleep(3)
    price = ticker.marketPrice()
    if price != price:
        price = ticker.close
    logging.info(f"📊 {SYMBOL}: ${price:.2f}")

    # Get chain — use ONLY strikes from the chain (they're the valid ones)
    chains = ib.reqSecDefOptParams(SYMBOL, "", "STK", stock.conId)
    chain = next(c for c in chains if c.exchange == "SMART")

    today = datetime.now().strftime("%Y%m%d")
    expiries = [e for e in sorted(chain.expirations) if e > today][:MAX_EXPIRATIONS]

    # Use ONLY strikes that exist in the chain AND are near ATM
    valid_strikes = sorted([
        s for s in chain.strikes
        if price * (1 - STRIKE_RANGE) <= s <= price * (1 + STRIKE_RANGE)
    ])

    logging.info(f"📅 Expiries: {expiries}")
    logging.info(f"💰 Valid strikes near ${price:.0f}: {valid_strikes}")

    # Build contracts using only valid chain strikes
    all_contracts = []
    for exp in expiries:
        for strike in valid_strikes:
            for right in ("C", "P"):
                all_contracts.append(Option(SYMBOL, exp, strike, right, "SMART"))

    logging.info(f"🔍 Qualifying {len(all_contracts)} options...")
    ib.qualifyContracts(*all_contracts)
    valid = [c for c in all_contracts if c.conId > 0]
    logging.info(f"✅ {len(valid)} valid contracts (filtered {len(all_contracts) - len(valid)} invalid)")

    found = 0
    has_greeks = 0

    for i in range(0, len(valid), BATCH_SIZE):
        batch = valid[i:i + BATCH_SIZE]
        tickers = ib.reqTickers(*batch)
        ib.sleep(5)

        for t in tickers:
            if not t.modelGreeks or not t.modelGreeks.delta:
                continue

            has_greeks += 1
            delta = abs(t.modelGreeks.delta)
            iv = t.modelGreeks.impliedVol or 0

            bid = t.bid if t.bid and t.bid > 0 else 0
            ask = t.ask if t.ask and t.ask > 0 else 0
            mid = (bid + ask) / 2 if bid and ask else 0
            opt_price = t.last if (t.last and t.last > 0) else mid or (t.close if t.close else 0)

            if opt_price <= 0:
                continue

            vol = int(t.volume) if t.volume and t.volume >= 0 else 0
            oi_val = t.callOpenInterest if t.contract.right == "C" else t.putOpenInterest
            oi = int(oi_val) if oi_val and oi_val >= 0 else 0

            right_str = "CALL" if t.contract.right == "C" else "PUT"
            exp_str = t.contract.lastTradeDateOrContractMonth

            if DEBUG:
                match = "✅" if (DELTA_MIN <= delta <= DELTA_MAX and vol >= MIN_VOL and oi >= MIN_OI) else "  "
                logging.info(f"  {match} {exp_str} ${t.contract.strike} {right_str:4} | ${opt_price:.2f} Δ{delta:.2f} IV:{iv:.1%} Vol:{vol} OI:{oi}")

            key = f"{exp_str}_{t.contract.strike}_{t.contract.right}"

            if (DELTA_MIN <= delta <= DELTA_MAX
                    and vol >= MIN_VOL
                    and oi >= MIN_OI
                    and key not in sent_alerts):

                send_alert({
                    "symbol": SYMBOL,
                    "expiry": exp_str,
                    "strike": t.contract.strike,
                    "right": right_str,
                    "price": opt_price,
                    "delta": delta,
                    "iv": iv,
                    "vol": vol,
                    "oi": oi,
                })
                sent_alerts.add(key)
                found += 1

        # Cancel tickers properly
        for t in tickers:
            try:
                ib.cancelMktData(t.contract)
            except:
                pass

    ib.disconnect()
    logging.info(f"📊 Greeks:{has_greeks} Alerts:{found}")

    if found > 0 or has_greeks > 0:
        no_data_alert_sent = False
    else:
        logging.warning("⚠️  No Greeks data")
        if not no_data_alert_sent:
            send_discord("⚠️ No Greeks data available")
            no_data_alert_sent = True


if __name__ == "__main__":
    logging.info(f"🚀 IBKR Options Scanner — {SYMBOL}")
    logging.info(f"📊 Delta: {DELTA_MIN}-{DELTA_MAX} | Vol:{MIN_VOL} OI:{MIN_OI}\n")

    if WEBHOOK_URL:
        send_discord(
            f"🚀 **Scanner Started — {SYMBOL}**\n"
            f"Delta: {DELTA_MIN}-{DELTA_MAX}\n"
            f"Min Vol/OI: {MIN_VOL}/{MIN_OI}\n"
            f"Interval: {SCAN_INTERVAL}s\n"
            f"⏰ {datetime.now().strftime('%I:%M:%S %p')}"
        )

    while True:
        try:
            scan()
        except KeyboardInterrupt:
            logging.info("\n👋 Stopped")
            if WEBHOOK_URL:
                send_discord("⏹️ **Scanner Stopped**")
            break
        except Exception as e:
            logging.error(f"Error: {e}")

        logging.info(f"Next scan in {SCAN_INTERVAL}s\n")
        time.sleep(SCAN_INTERVAL)