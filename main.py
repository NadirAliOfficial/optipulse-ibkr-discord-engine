import time
from datetime import datetime
from typing import Dict, Set, Optional
import logging
from dotenv import load_dotenv
import os

load_dotenv()

try:
    from ib_insync import IB, Stock, Option, util
    import requests
except ImportError:
    print("ERROR: Required packages not installed!")
    print("Run: pip install ib_insync requests python-dotenv")
    exit(1)

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Interactive Brokers Connection
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7497=TWS live, 7496=TWS paper, 4002=Gateway paper, 4001=Gateway live
IB_CLIENT_ID = 1

# Discord Integration
DISCORD_WEBHOOK_URL = os.getenv('WEBHOOK_URL', 'YOUR_DISCORD_WEBHOOK_URL_HERE')

# Stock Symbol
SYMBOL = 'SPY'  # Stock to scan
EXCHANGE = 'SMART'

# FALLBACK PRICES - UPDATE THESE TO CURRENT PRICES!
FALLBACK_PRICES = {
    'AAPL': 235.0,
    'TSLA': 415.0,
    'SHOP': 82.0,
    'SPY': 600.0,
    'QQQ': 520.0,
    'NVDA': 140.0,
    'MSFT': 445.0,
}

# Delta Filter
DELTA_MIN = 0.35
DELTA_MAX = 0.55

# Spread Filter
MAX_SPREAD_PERCENT = 7.0

# Liquidity Filters
MIN_VOLUME = 10
MIN_OPEN_INTEREST = 50

# Scanning Behavior
SCAN_INTERVAL_SECONDS = 60
RECONNECT_DELAY_SECONDS = 5

# Logging
LOG_LEVEL = logging.INFO

# Market Data Settings
USE_DELAYED_DATA = True
STRIKE_FILTER_PERCENT = 0.20
MAX_EXPIRATIONS = 3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_strikes_near_money(strikes: list, current_price: float, range_pct: float = 0.20) -> list:
    """Keep only strikes within ±X% of current stock price to reduce API calls"""
    if not strikes or not current_price or current_price <= 0:
        return strikes[:15] if strikes else []
    
    min_strike = current_price * (1 - range_pct)
    max_strike = current_price * (1 + range_pct)
    filtered = [s for s in strikes if min_strike <= s <= max_strike]
    
    if len(filtered) < 5:
        return sorted(strikes, key=lambda x: abs(x - current_price))[:15]
    
    return filtered


def get_stock_price_safe(ib_connection, symbol: str, fallback: float) -> float:
    """Get current stock price with multiple fallback methods"""
    try:
        stock = Stock(symbol, EXCHANGE, 'USD')
        ib_connection.qualifyContracts(stock)
        ticker = ib_connection.reqMktData(stock, '', True, False)
        
        ib_connection.sleep(4)
        
        price_sources = [
            ('last', ticker.last),
            ('close', ticker.close),
            ('bid', ticker.bid),
            ('ask', ticker.ask),
        ]
        
        for source_name, price_value in price_sources:
            if price_value and price_value > 0 and str(price_value) != 'nan':
                price = float(price_value)
                logging.info(f"✅ Got stock price from '{source_name}': ${price:.2f}")
                ib_connection.cancelMktData(stock)
                return price
        
        if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
            midpoint = (float(ticker.bid) + float(ticker.ask)) / 2
            logging.info(f"✅ Got stock price from bid/ask midpoint: ${midpoint:.2f}")
            ib_connection.cancelMktData(stock)
            return midpoint
        
        ib_connection.cancelMktData(stock)
        
    except Exception as e:
        logging.warning(f"Error fetching stock price: {e}")
    
    logging.warning(f"⚠️  Could not fetch price dynamically, using fallback: ${fallback:.2f}")
    return fallback

# ============================================================================
# ALERT TRACKING
# ============================================================================

class AlertTracker:
    """Tracks sent alerts to prevent duplicate notifications"""
    
    def __init__(self):
        self.sent_alerts: Set[str] = set()
        self.last_data: Dict[str, Dict] = {}
    
    def generate_key(self, contract_data: Dict) -> str:
        return f"{contract_data['symbol']}_{contract_data['expiry']}_{contract_data['strike']}_{contract_data['right']}"
    
    def should_send_alert(self, contract_data: Dict) -> bool:
        key = self.generate_key(contract_data)
        
        if key not in self.sent_alerts:
            self.sent_alerts.add(key)
            self.last_data[key] = contract_data
            return True
        
        old_data = self.last_data.get(key, {})
        old_spread = old_data.get('spread_percent', float('inf'))
        new_spread = contract_data.get('spread_percent', float('inf'))
        
        if old_spread - new_spread > 1.0:
            self.last_data[key] = contract_data
            return True
        
        return False

# ============================================================================
# DISCORD INTEGRATION
# ============================================================================

def send_discord_alert(contract_data: Dict, webhook_url: str) -> bool:
    """Send formatted alert to Discord"""
    
    if not webhook_url or webhook_url == 'YOUR_DISCORD_WEBHOOK_URL_HERE':
        logging.warning("⚠️  Discord webhook not configured - skipping send")
        return False
    
    message = f"""
**🎯 New Option Alert**

**Stock:** {contract_data['symbol']}
**Type:** {contract_data['right']}
**Expiry:** {contract_data['expiry']}
**Strike:** ${contract_data['strike']:.2f}

**Pricing:**
• Bid: ${contract_data['bid']:.2f}
• Ask: ${contract_data['ask']:.2f}
• Spread: {contract_data['spread_percent']:.2f}%

**Greeks & Volume:**
• Delta: {contract_data['delta']:.3f}
• Volume: {contract_data['volume']:,}
• Open Interest: {contract_data['open_interest']:,}

*Scanned at {contract_data['timestamp']}*
    """.strip()
    
    payload = {
        "content": message,
        "username": "IBKR Options Scanner"
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 204:
            logging.info(f"✅ Alert sent to Discord: {contract_data['symbol']} {contract_data['strike']} {contract_data['right']}")
            return True
        else:
            logging.error(f"❌ Discord webhook failed: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Failed to send Discord alert: {e}")
        return False

# ============================================================================
# OPTIONS FILTERING
# ============================================================================

def calculate_spread_percent(bid: float, ask: float) -> Optional[float]:
    if ask <= 0 or bid < 0:
        return None
    spread = ((ask - bid) / ask) * 100
    return spread

def passes_delta_filter(delta: Optional[float], right: str) -> bool:
    if delta is None:
        return False
    abs_delta = abs(delta)
    return DELTA_MIN <= abs_delta <= DELTA_MAX

def passes_spread_filter(bid: float, ask: float) -> bool:
    if bid <= 0 or ask <= 0:
        return False
    spread_percent = calculate_spread_percent(bid, ask)
    if spread_percent is None:
        return False
    return spread_percent <= MAX_SPREAD_PERCENT

def passes_liquidity_filter(volume: int, open_interest: int) -> bool:
    return volume >= MIN_VOLUME and open_interest >= MIN_OPEN_INTEREST

# ============================================================================
# IBKR CONNECTION & DATA RETRIEVAL
# ============================================================================

class IBKRScanner:
    """Manages IBKR connection and options scanning"""
    
    def __init__(self):
        self.ib = IB()
        self.alert_tracker = AlertTracker()
        self.is_connected = False
        
        logging.basicConfig(
            level=LOG_LEVEL,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if LOG_LEVEL != logging.DEBUG:
            logging.getLogger('ib_insync').setLevel(logging.WARNING)
    
    def connect(self) -> bool:
        """Connect to Interactive Brokers"""
        try:
            if self.is_connected:
                return True
            
            logging.info(f"🔌 Connecting to IB Gateway/TWS at {IB_HOST}:{IB_PORT}...")
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=True)
            
            # REQUEST DELAYED MARKET DATA (Critical fix!)
            self.ib.reqMarketDataType(3)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
            
            self.is_connected = True
            logging.info("✅ Connected to IBKR successfully!")
            logging.info("📊 Delayed market data mode enabled (Type 3)")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to connect to IBKR: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        if self.is_connected:
            try:
                self.ib.disconnect()
                logging.info("🔌 Disconnected from IBKR")
            except:
                pass
            self.is_connected = False
    
    def ensure_connection(self) -> bool:
        if not self.ib.isConnected():
            logging.warning("⚠️  Connection lost, attempting to reconnect...")
            self.is_connected = False
            time.sleep(RECONNECT_DELAY_SECONDS)
            return self.connect()
        return True
    
    def get_option_chains(self, symbol: str) -> list:
        try:
            stock = Stock(symbol, EXCHANGE, 'USD')
            self.ib.qualifyContracts(stock)
            
            chains = self.ib.reqSecDefOptParams(
                stock.symbol,
                '',
                stock.secType,
                stock.conId
            )
            
            if not chains:
                logging.warning(f"⚠️  No option chains found for {symbol}")
                return []
            
            logging.info(f"📊 Found {len(chains)} option chain(s) for {symbol}")
            return chains
            
        except Exception as e:
            logging.error(f"❌ Failed to get option chains: {e}")
            return []
    
    def create_option_contracts(self, chains: list, symbol: str) -> list:
        """Create option contracts from chains (FIXED VERSION)"""
        contracts = []
        
        if not chains:
            logging.error("❌ No chains available")
            return []
        
        # Get fallback price for this symbol
        fallback = FALLBACK_PRICES.get(symbol, 100.0)
        
        # Get current stock price with fallbacks
        current_price = get_stock_price_safe(self.ib, symbol, fallback)
        
        logging.info(f"💰 Current {symbol} price: ${current_price:.2f}")
        
        # Use only the first chain
        chain = chains[0]
        logging.info(f"📊 Using 1 chain (ignoring {len(chains)-1} duplicate chains)")
        
        all_expirations = sorted(chain.expirations)
        expirations = all_expirations[:MAX_EXPIRATIONS]
        
        all_strikes = sorted(chain.strikes)
        strikes = filter_strikes_near_money(all_strikes, current_price, STRIKE_FILTER_PERCENT)
        
        logging.info(f"📅 Using {len(expirations)} expirations (of {len(all_expirations)}) with {len(strikes)} strikes (of {len(all_strikes)})")
        logging.info(f"💡 Strike range: ${min(strikes):.2f} - ${max(strikes):.2f}")
        
        for expiry in expirations:
            for strike in strikes:
                call = Option(symbol, expiry, strike, 'C', EXCHANGE)
                contracts.append(call)
                
                put = Option(symbol, expiry, strike, 'P', EXCHANGE)
                contracts.append(put)
        
        logging.info(f"📝 Created {len(contracts)} option contracts (reduced from potential 72,000+)")
        
        if len(contracts) > 500:
            logging.error(f"⚠️  WARNING: Created {len(contracts)} contracts - limiting to 300")
            contracts = contracts[:300]
        
        return contracts
    
    def get_market_data(self, contracts: list) -> list:
        """Get market data for option contracts"""
        BATCH_SIZE = 25
        qualified_options = []
        
        total_batches = (len(contracts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logging.info(f"📊 Requesting market data for {len(contracts)} contracts in {total_batches} batches...")
        
        for i in range(0, len(contracts), BATCH_SIZE):
            batch = contracts[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            logging.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} contracts)...")
            
            try:
                qualified = self.ib.qualifyContracts(*batch)
                
                for contract in qualified:
                    ticker = self.ib.reqMktData(contract, '', True, False)
                
                self.ib.sleep(3)
                
                for contract in qualified:
                    ticker = self.ib.ticker(contract)
                    if ticker:
                        qualified_options.append(ticker)
                
                for contract in qualified:
                    try:
                        self.ib.cancelMktData(contract)
                    except:
                        pass
                
            except Exception as e:
                logging.error(f"❌ Error processing batch {batch_num}: {e}")
                continue
            
            if i + BATCH_SIZE < len(contracts):
                time.sleep(1)
        
        logging.info(f"✅ Retrieved data for {len(qualified_options)} options")
        return qualified_options
    
    def filter_and_alert(self, tickers: list):
        """Filter options and send alerts"""
        alerts_sent = 0
        options_processed = 0
        
        for ticker in tickers:
            options_processed += 1
            
            try:
                contract = ticker.contract
                
                bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
                ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
                volume = ticker.volume if ticker.volume else 0
                
                delta = None
                if ticker.modelGreeks:
                    delta = ticker.modelGreeks.delta
                
                open_interest = 0
                if hasattr(ticker, 'callOpenInterest') and contract.right == 'C':
                    open_interest = ticker.callOpenInterest or 0
                elif hasattr(ticker, 'putOpenInterest') and contract.right == 'P':
                    open_interest = ticker.putOpenInterest or 0
                
                if not passes_delta_filter(delta, contract.right):
                    continue
                
                if not passes_spread_filter(bid, ask):
                    continue
                
                if not passes_liquidity_filter(volume, open_interest):
                    continue
                
                spread_percent = calculate_spread_percent(bid, ask)
                
                contract_data = {
                    'symbol': contract.symbol,
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'strike': contract.strike,
                    'right': 'CALL' if contract.right == 'C' else 'PUT',
                    'bid': bid,
                    'ask': ask,
                    'spread_percent': spread_percent,
                    'delta': delta,
                    'volume': int(volume),
                    'open_interest': int(open_interest),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                if self.alert_tracker.should_send_alert(contract_data):
                    if send_discord_alert(contract_data, DISCORD_WEBHOOK_URL):
                        alerts_sent += 1
                        time.sleep(1)
                
            except Exception as e:
                logging.debug(f"Error processing ticker: {e}")
                continue
        
        logging.info(f"📊 Scan complete: {options_processed} options processed, {alerts_sent} alerts sent")
    
    def scan_once(self):
        """Perform one complete scan"""
        logging.info(f"🔍 Starting scan for {SYMBOL}...")
        
        if not self.ensure_connection():
            logging.error("❌ Cannot scan - not connected to IBKR")
            return
        
        chains = self.get_option_chains(SYMBOL)
        if not chains:
            return
        
        contracts = self.create_option_contracts(chains, SYMBOL)
        if not contracts:
            return
        
        tickers = self.get_market_data(contracts)
        if not tickers:
            logging.warning("⚠️  No market data retrieved")
            return
        
        self.filter_and_alert(tickers)
    
    def run_continuous(self):
        """Run scanner continuously"""
        logging.info("🚀 Starting IBKR Options Scanner (WITH DELAYED DATA)")
        logging.info(f"📊 Symbol: {SYMBOL}")
        logging.info(f"🎯 Delta range: {DELTA_MIN} - {DELTA_MAX}")
        logging.info(f"💰 Max spread: {MAX_SPREAD_PERCENT}%")
        logging.info(f"📈 Min volume: {MIN_VOLUME}, Min OI: {MIN_OPEN_INTEREST}")
        logging.info(f"⏱️  Scan interval: {SCAN_INTERVAL_SECONDS} seconds")
        logging.info("=" * 70)
        
        if not self.connect():
            logging.error("❌ Initial connection failed. Please check IB Gateway/TWS is running.")
            return
        
        try:
            while True:
                try:
                    self.scan_once()
                except Exception as e:
                    logging.error(f"❌ Error during scan: {e}")
                    import traceback
                    traceback.print_exc()
                
                logging.info(f"⏳ Waiting {SCAN_INTERVAL_SECONDS} seconds until next scan...\n")
                time.sleep(SCAN_INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            logging.info("\n⚠️  Shutdown requested by user")
        finally:
            self.disconnect()
            logging.info("👋 Scanner stopped")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == 'YOUR_DISCORD_WEBHOOK_URL_HERE':
        print("\n" + "=" * 70)
        print("⚠️  WARNING: Discord webhook URL not configured!")
        print("=" * 70)
        print("The scanner will run but alerts won't be sent to Discord.")
        print("To fix: Set WEBHOOK_URL in .env file or edit DISCORD_WEBHOOK_URL")
        print("=" * 70 + "\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    print("\n" + "=" * 70)
    print("🚀 IBKR Options Scanner - WITH DELAYED DATA FIX")
    print("=" * 70)
    print(f"Port: {IB_PORT} ({'TWS' if IB_PORT in [7496, 7497] else 'Gateway'})")
    print(f"Symbol: {SYMBOL}")
    print(f"Fallback Price: ${FALLBACK_PRICES.get(SYMBOL, 100.0)}")
    print(f"Delayed Data: ENABLED (reqMarketDataType=3)")
    print("=" * 70 + "\n")
    
    scanner = IBKRScanner()
    scanner.run_continuous()

if __name__ == "__main__":
    main()