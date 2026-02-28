import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import pytz

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from ib_insync import IB, Stock, Option

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — PHASE 2.1 CLIENT TUNING
# ══════════════════════════════════════════════════════════════════════════════

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
SYMBOL = os.getenv("SYMBOL", "QQQ")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

TIMEFRAMES = {
    "1m":  {"bar_size": "1 min",   "duration": "7200 S",  "bars_needed": 25},
    "5m":  {"bar_size": "5 mins",  "duration": "28800 S", "bars_needed": 25},
    "15m": {"bar_size": "15 mins", "duration": "86400 S", "bars_needed": 25},
}

# Indicators
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD, BB_STD = 20, 2.0
BB_SQUEEZE_THRESHOLD = 0.04
ADX_THRESHOLD = 20
VOLUME_LOOKBACK = 20
VOLUME_EXPANSION_MULT = 1.5

RSI_CALL_MIN, RSI_CALL_MAX = 40, 70
RSI_PUT_MIN, RSI_PUT_MAX = 30, 60

# ═══ PHASE 2.1: Tuned Scoring ═══
MIN_SCORE = 5
TOTAL_CHECKS = 7
REQUIRE_MULTI_TF_STRICT = True      # Keep 5m + 15m check
VOLUME_IS_HARD_GATE = False         # CHANGED: Volume warns, doesn't block
MULTI_TF_ALLOW_15M_LAG = True       # NEW: 15m can be neutral (not opposite)

# ═══ PHASE 2.1: Enhanced Options Filters ═══
DELTA_MIN, DELTA_MAX = 0.40, 0.60
STRIKE_RANGE = 0.02
MAX_SPREAD_PERCENT = 0.15
MIN_OPTION_VOLUME = 50
MIN_OPEN_INTEREST = 500
MIN_DTE = 0
MAX_DTE = 2

MAX_EXPIRATIONS = 2
BATCH_SIZE = 40

# ═══ PHASE 2.1: Alert Throttling ═══
ALERT_COOLDOWN_SECONDS = 600
ALLOW_DIRECTION_FLIP = True
ALERT_ON_THRESHOLD_CROSS = False    # CHANGED: Alert on first qualify, not just crosses
DAILY_RESET_AT_OPEN = True          # NEW: Reset state at 9:30 ET

SCAN_INTERVAL = 60
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30     # Changed to actual market hours
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 16, 0

SHADOW_LOG_DIR = "shadow_logs"
CHART_DIR = "charts"

os.makedirs(SHADOW_LOG_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("optipulse")
logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
logging.getLogger("ib_insync.client").setLevel(logging.WARNING)

EASTERN = pytz.timezone('US/Eastern')

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df, period=14):
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = compute_atr(df, period)
    plus_di = 100 * compute_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * compute_ema(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return compute_ema(dx, period)

def compute_bb_bandwidth(series, period=20, std=2.0):
    sma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    return ((sma + std * sd) - (sma - std * sd)) / sma.replace(0, np.nan)

def enrich_df(df):
    df = df.copy()
    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)
    df["vwap"] = compute_vwap(df)
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["adx"] = compute_adx(df, ADX_PERIOD)
    df["bb_bw"] = compute_bb_bandwidth(df["close"], BB_PERIOD, BB_STD)
    avg_vol = df["volume"].rolling(VOLUME_LOOKBACK).mean()
    df["vol_expansion"] = df["volume"] > (avg_vol * VOLUME_EXPANSION_MULT)
    df["candle_range"] = df["high"] - df["low"]
    return df

# ══════════════════════════════════════════════════════════════════════════════
# BAR COLLECTOR (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def bars_to_df(bars):
    if not bars:
        return pd.DataFrame()
    data = [{"date": b.date, "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def fetch_bars(ib, stock, bar_size, duration):
    try:
        bars = ib.reqHistoricalData(
            stock, endDateTime="", durationStr=duration,
            barSizeSetting=bar_size, whatToShow="TRADES",
            useRTH=False, formatDate=1, timeout=30,
        )
        return bars_to_df(bars)
    except Exception as e:
        log.error(f"Error fetching {bar_size}: {e}")
        return pd.DataFrame()

def collect_all_timeframes(ib, stock):
    result = {}
    for tf, cfg in TIMEFRAMES.items():
        log.info(f"  Fetching {tf}...")
        df = fetch_bars(ib, stock, cfg["bar_size"], cfg["duration"])
        if df.empty or len(df) < cfg["bars_needed"]:
            log.warning(f"  {tf}: insufficient ({len(df)} bars)")
            result[tf] = None
            continue
        df = enrich_df(df)
        result[tf] = df
        log.info(f"  {tf}: {len(df)} bars, close=${df['close'].iloc[-1]:.2f}")
    return result

def get_current_price(ib, stock):
    [ticker] = ib.reqTickers(stock)
    ib.sleep(3)
    p = ticker.marketPrice()
    return p if p == p else ticker.close

# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — PHASE 2.1 TUNED
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    warning: bool = False  # NEW: For volume warnings

@dataclass
class SignalScore:
    direction: str
    score: int
    total: int
    checks: List[CheckResult]
    tf_details: Dict[str, str] = field(default_factory=dict)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    signal_candle_time: str = ""
    volume_confirmed: bool = True  # NEW: Track volume separately

    @property
    def passed_names(self):
        return [c.name for c in self.checks if c.passed]

    @property
    def qualifies(self):
        return self.score >= MIN_SCORE

    def summary(self):
        vol_warn = " [⚠️ VOL]" if not self.volume_confirmed else ""
        return f"{self.direction} {self.score}/{self.total}{vol_warn} [{' | '.join(self.passed_names)}]"

def _score_single_tf(df):
    last = df.iloc[-1]
    bullish = last["ema_fast"] > last["ema_slow"]
    direction = "CALL" if bullish else "PUT"
    checks = {}

    checks["ema"] = {"passed": True, "detail": f"EMA9={last['ema_fast']:.2f} vs EMA21={last['ema_slow']:.2f}"}

    if pd.notna(last["vwap"]):
        vwap_ok = last["close"] > last["vwap"] if bullish else last["close"] < last["vwap"]
        checks["vwap"] = {"passed": vwap_ok, "detail": f"Close={last['close']:.2f} vs VWAP={last['vwap']:.2f}"}
    else:
        checks["vwap"] = {"passed": False, "detail": "VWAP N/A"}

    rsi = last["rsi"]
    if bullish:
        rsi_ok = RSI_CALL_MIN <= rsi <= RSI_CALL_MAX
    else:
        rsi_ok = RSI_PUT_MIN <= rsi <= RSI_PUT_MAX
    checks["rsi"] = {"passed": rsi_ok, "detail": f"RSI={rsi:.1f}"}

    # PHASE 2.1: Volume as warning, not blocker
    vol_pass = bool(last["vol_expansion"])
    checks["volume"] = {"passed": vol_pass, "detail": f"Vol={last['volume']:.0f}", "warning": not vol_pass}

    atr_ok = last["candle_range"] > 0.5 * last["atr"] if pd.notna(last["atr"]) else False
    checks["atr"] = {"passed": atr_ok,
                      "detail": f"Range={last['candle_range']:.2f} vs 0.5*ATR={0.5 * last['atr']:.2f}" if pd.notna(last["atr"]) else "N/A"}

    adx_ok = last["adx"] > ADX_THRESHOLD if pd.notna(last["adx"]) else False
    bb_ok = last["bb_bw"] > BB_SQUEEZE_THRESHOLD if pd.notna(last["bb_bw"]) else False
    checks["chop"] = {"passed": adx_ok or bb_ok,
                       "detail": f"ADX={last['adx']:.1f} BB={last['bb_bw']:.3f}" if pd.notna(last["adx"]) else "N/A"}

    return {"direction": direction, "checks": checks}

def score_multi_timeframe(tf_data):
    tf_scores = {}
    tf_details = {}
    
    for tf, df in tf_data.items():
        if df is None or len(df) < 25:
            continue
        tf_scores[tf] = _score_single_tf(df)
        s = tf_scores[tf]
        passed = sum(1 for c in s["checks"].values() if c["passed"])
        tf_details[tf] = f"{s['direction']} {passed}/6"

    if not tf_scores:
        return SignalScore("NONE", 0, TOTAL_CHECKS, [])

    # PHASE 2.1: Relaxed Multi-TF (15m can lag)
    if REQUIRE_MULTI_TF_STRICT:
        if "5m" not in tf_scores:
            log.warning("⚠️  Need 5m timeframe")
            return SignalScore("NONE", 0, TOTAL_CHECKS, [])
        
        dir_5m = tf_scores["5m"]["direction"]
        
        # PHASE 2.1: Check 15m is not opposite (can be same or missing)
        if "15m" in tf_scores and MULTI_TF_ALLOW_15M_LAG:
            dir_15m = tf_scores["15m"]["direction"]
            if dir_15m != dir_5m:
                log.warning(f"⚠️  TF opposite: 5m={dir_5m}, 15m={dir_15m} (BLOCKED)")
                return SignalScore("NONE", 0, TOTAL_CHECKS, [])
        
        direction = dir_5m
        primary = tf_scores["5m"]
    else:
        primary = tf_scores.get("5m") or list(tf_scores.values())[0]
        direction = primary["direction"]

    checks = []
    volume_confirmed = True

    # PHASE 2.1: Volume doesn't block, just warns
    vol_check = primary["checks"]["volume"]
    if not vol_check["passed"]:
        volume_confirmed = False

    # 1-5: from primary TF
    for name, label in [("ema", "EMA"), ("vwap", "VWAP"), ("rsi", "RSI"),
                         ("volume", "Volume"), ("atr", "ATR")]:
        c = primary["checks"][name]
        is_warning = c.get("warning", False)
        checks.append(CheckResult(label, c["passed"], c["detail"], is_warning))

    # 6: Multi-TF agreement
    dirs = [s["direction"] for s in tf_scores.values()]
    
    if REQUIRE_MULTI_TF_STRICT and MULTI_TF_ALLOW_15M_LAG:
        agreement = len([d for d in dirs if d == direction])
        tf_check_pass = True  # Already checked above
        tf_detail = f"5m={direction}, 15m not opposite ({agreement}/3 TFs)"
    else:
        agreement = dirs.count(direction)
        tf_check_pass = agreement >= 2
        tf_detail = f"{agreement}/{len(dirs)} TFs agree"
    
    checks.append(CheckResult("Multi-TF", tf_check_pass, tf_detail))

    # 7: Chop filter
    chop_pass = any(s["checks"]["chop"]["passed"] for s in tf_scores.values())
    chop_detail = " | ".join(f"{tf}:{s['checks']['chop']['detail']}" for tf, s in tf_scores.items())
    checks.append(CheckResult("Chop", chop_pass, chop_detail))

    score = sum(1 for c in checks if c.passed)
    
    # Entry/stop calculation
    primary_df = None
    if tf_data.get("5m") is not None:
        primary_df = tf_data.get("5m")
    elif tf_data.get("1m") is not None:
        primary_df = tf_data.get("1m")
    else:
        primary_df = [v for v in tf_data.values() if v is not None][0]
    
    last_candle = primary_df.iloc[-1]
    entry_price = last_candle["close"]
    
    atr = last_candle["atr"] if pd.notna(last_candle["atr"]) else (entry_price * 0.01)
    if direction == "CALL":
        stop_loss = entry_price - (1.5 * atr)
    else:
        stop_loss = entry_price + (1.5 * atr)
    
    signal_time = last_candle.name.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
    
    return SignalScore(
        direction, score, TOTAL_CHECKS, checks, tf_details,
        entry_price, stop_loss, signal_time, volume_confirmed
    )

# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT PICKER (unchanged from Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContractPick:
    expiry: str
    strike: float
    right: str
    price: float
    bid: float
    ask: float
    spread: float
    spread_pct: float
    delta: float
    iv: float
    volume: int
    oi: int
    dte: int
    rank_score: float

    def display_right(self):
        return "CALL" if self.right == "C" else "PUT"

    def line(self):
        return (f"${self.strike} {self.display_right()} {self.expiry} ({self.dte}DTE) | "
                f"${self.price:.2f} Δ{self.delta:.2f} IV:{self.iv:.0%} | "
                f"Vol:{self.volume} OI:{self.oi} | "
                f"Spread:${self.spread:.2f}({self.spread_pct:.1%})")

def calculate_dte(expiry_str):
    try:
        expiry = datetime.strptime(expiry_str, "%Y%m%d")
        now = datetime.now()
        return (expiry - now).days
    except:
        return 999

def pick_top_contracts(ib, stock, price, direction, max_picks=3):
    """PHASE 2.2: Returns (contracts_list, error_message)"""
    right = "C" if direction == "CALL" else "P"
    chains = ib.reqSecDefOptParams(stock.symbol, "", "STK", stock.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), None)
    if not chain:
        return [], "No SMART chain found"

    # PHASE 2.2: Enhanced DTE logging with timezone
    now_et = datetime.now(EASTERN)
    today = now_et.strftime("%Y%m%d")
    
    log.info(f"  📅 Today (ET): {today} | Time: {now_et:%Y-%m-%d %H:%M:%S %Z}")
    
    # Log what IBKR returned
    all_expiries = sorted(chain.expirations)
    log.info(f"  📊 IBKR returned {len(all_expiries)} expirations (showing first 5):")
    for exp in all_expiries[:5]:
        dte = calculate_dte(exp)
        status = "✅" if MIN_DTE <= dte <= MAX_DTE else "❌"
        log.info(f"     {status} {exp} -> {dte} DTE")
    
    expiries = []
    for e in all_expiries:
        if e <= today:
            continue
        dte = calculate_dte(e)
        if MIN_DTE <= dte <= MAX_DTE:
            expiries.append(e)
    
    expiries = expiries[:MAX_EXPIRATIONS]
    
    # PHASE 2.2: Fallback to 0-5 DTE if primary range empty
    if not expiries:
        log.warning(f"  ⚠️  No expiries in {MIN_DTE}-{MAX_DTE} DTE, trying fallback (0-5 DTE)...")
        for e in all_expiries:
            if e <= today:
                continue
            dte = calculate_dte(e)
            if 0 <= dte <= 5:
                expiries.append(e)
        expiries = expiries[:MAX_EXPIRATIONS]
        
        if expiries:
            log.info(f"  ✅ Fallback success: {len(expiries)} expiries (0-5 DTE)")
        else:
            return [], f"No expiries in 0-{MAX_DTE} DTE or 0-5 DTE fallback"
    
    log.info(f"  Using {len(expiries)} expiries: {', '.join(f'{calculate_dte(e)}DTE' for e in expiries)}")
    
    strikes = sorted([s for s in chain.strikes
                      if price * (1 - STRIKE_RANGE) <= s <= price * (1 + STRIKE_RANGE)])
    
    if not strikes:
        return [], f"No strikes within ±{STRIKE_RANGE*100:.0f}% of ${price:.2f}"
    
    log.info(f"  Strike range: ${min(strikes):.2f} - ${max(strikes):.2f} ({len(strikes)} strikes)")

    contracts = [Option(stock.symbol, exp, s, right, "SMART") for exp in expiries for s in strikes]
    log.info(f"  Qualifying {len(contracts)} {direction} contracts...")
    ib.qualifyContracts(*contracts)
    valid = [c for c in contracts if c.conId > 0]
    
    if not valid:
        return [], "No valid contracts after qualification"

    picks = []
    for i in range(0, len(valid), BATCH_SIZE):
        batch = valid[i:i + BATCH_SIZE]
        tickers = ib.reqTickers(*batch)
        ib.sleep(4)

        for t in tickers:
            if not t.modelGreeks or not t.modelGreeks.delta:
                continue
            
            delta = abs(t.modelGreeks.delta)
            if not (DELTA_MIN <= delta <= DELTA_MAX):
                continue

            iv = t.modelGreeks.impliedVol or 0
            bid = t.bid if t.bid and t.bid > 0 else 0
            ask = t.ask if t.ask and t.ask > 0 else 0

            if bid > 0 and ask > 0:
                price_opt = (bid + ask) / 2
                spread = ask - bid
            else:
                price_opt = t.last if (t.last and t.last > 0) else (t.close if t.close else 0)
                if price_opt <= 0:
                    continue
                spread = 0

            spread_pct = spread / price_opt if price_opt > 0 else 1.0
            
            if spread_pct > MAX_SPREAD_PERCENT:
                continue
            
            vol = int(t.volume) if t.volume and t.volume >= 0 else 0
            oi_val = t.callOpenInterest if right == "C" else t.putOpenInterest
            oi = int(oi_val) if oi_val and oi_val >= 0 else 0

            if vol < MIN_OPTION_VOLUME or oi < MIN_OPEN_INTEREST:
                continue
            
            dte = calculate_dte(t.contract.lastTradeDateOrContractMonth)

            d_score = abs(delta - 0.50) / 0.20
            s_score = min(spread_pct / MAX_SPREAD_PERCENT, 1.0)
            l_score = max(0, 1.0 - ((vol + oi) / 2000))
            rank = 0.30 * d_score + 0.30 * s_score + 0.40 * l_score

            picks.append(ContractPick(
                t.contract.lastTradeDateOrContractMonth, t.contract.strike,
                right, price_opt, bid, ask, spread, spread_pct,
                delta, iv, vol, oi, dte, rank,
            ))

        for t in tickers:
            try: 
                ib.cancelMktData(t.contract)
            except: 
                pass

    if not picks:
        return [], "No contracts passed filters (spread/volume/OI)"
    
    picks.sort(key=lambda p: p.rank_score)
    top = picks[:max_picks]
    
    for i, p in enumerate(top):
        log.info(f"  #{i+1}: {p.line()}")
    
    return top, None  # Success, no error

# ══════════════════════════════════════════════════════════════════════════════
# CHARTING (unchanged from Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#1a1a2e", "panel": "#16213e", "text": "#e0e0e0", "grid": "#2a2a4a",
    "bull": "#00e676", "bear": "#ff1744", "ema_fast": "#ffab00", "ema_slow": "#448aff",
    "vwap": "#e040fb", "rsi": "#26c6da", "vol_bar": "#546e7a", "vol_hi": "#ffab00",
    "trigger": "#ffd600", "entry": "#00ff00", "stop": "#ff0000",
}

def generate_chart(tf_data, signal, contracts_text=""):
    primary = None
    for tf in ["5m", "1m", "15m"]:
        if tf in tf_data and tf_data[tf] is not None:
            primary = tf
            break
    if not primary:
        return None

    df = tf_data[primary].copy()
    direction = signal.direction
    score = signal.score
    total = signal.total
    passed_checks = signal.passed_names
    entry_price = signal.entry_price
    stop_loss = signal.stop_loss
    
    try:
        fig = plt.figure(figsize=(14, 10), facecolor=COLORS["bg"])
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        ax_p = fig.add_subplot(gs[0])
        ax_v = fig.add_subplot(gs[1], sharex=ax_p)
        ax_r = fig.add_subplot(gs[2], sharex=ax_p)

        for ax in [ax_p, ax_v, ax_r]:
            ax.set_facecolor(COLORS["panel"])
            ax.tick_params(colors=COLORS["text"], labelsize=8)
            ax.grid(True, alpha=0.15, color=COLORS["grid"])
            for sp in ax.spines.values():
                sp.set_color(COLORS["grid"])

        last_n = 50
        dp = df.tail(last_n).reset_index()
        n = len(dp)

        for i, row in dp.iterrows():
            c = COLORS["bull"] if row["close"] >= row["open"] else COLORS["bear"]
            ax_p.plot([i, i], [row["low"], row["high"]], color=c, linewidth=0.8)
            bot = min(row["open"], row["close"])
            h = max(abs(row["close"] - row["open"]), 0.001)
            ax_p.add_patch(plt.Rectangle((i - 0.35, bot), 0.7, h, facecolor=c, edgecolor=c, lw=0.5))

        x = range(n)
        ax_p.plot(x, dp["ema_fast"].values, color=COLORS["ema_fast"], lw=1.2, label=f"EMA{EMA_FAST}", alpha=0.9)
        ax_p.plot(x, dp["ema_slow"].values, color=COLORS["ema_slow"], lw=1.2, label=f"EMA{EMA_SLOW}", alpha=0.9)
        if "vwap" in dp.columns:
            ax_p.plot(x, dp["vwap"].values, color=COLORS["vwap"], lw=1.0, label="VWAP", alpha=0.7, ls="--")

        ti = n - 1
        ax_p.axhline(y=entry_price, color=COLORS["entry"], lw=2, alpha=0.7, ls="-", label=f"Entry: ${entry_price:.2f}")
        ax_p.axhline(y=stop_loss, color=COLORS["stop"], lw=2, alpha=0.7, ls="--", label=f"Stop: ${stop_loss:.2f}")
        ax_p.axvline(x=ti, color=COLORS["trigger"], lw=2, alpha=0.4, ls="--")
        
        arrow_emoji = "▲" if direction == "CALL" else "▼"
        ax_p.annotate(f"{arrow_emoji} ENTRY", xy=(ti, entry_price),
                       fontsize=10, fontweight="bold", color=COLORS["entry"], ha="center", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg"], ec=COLORS["entry"], alpha=0.9))

        ax_p.legend(loc="upper left", fontsize=8, facecolor=COLORS["panel"],
                     edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
        ax_p.set_ylabel("Price", color=COLORS["text"], fontsize=9)

        vc = [COLORS["vol_hi"] if dp.iloc[i].get("vol_expansion", False) else COLORS["vol_bar"] for i in range(n)]
        ax_v.bar(x, dp["volume"].values, color=vc, alpha=0.7, width=0.7)
        ax_v.set_ylabel("Vol", color=COLORS["text"], fontsize=9)

        ax_r.plot(x, dp["rsi"].values, color=COLORS["rsi"], lw=1.2)
        ax_r.axhline(70, color=COLORS["bear"], lw=0.7, alpha=0.5, ls="--")
        ax_r.axhline(30, color=COLORS["bull"], lw=0.7, alpha=0.5, ls="--")
        ax_r.axhline(50, color=COLORS["text"], lw=0.5, alpha=0.3, ls=":")
        ax_r.set_ylim(10, 90)
        ax_r.set_ylabel("RSI", color=COLORS["text"], fontsize=9)

        if "date" in dp.columns:
            ticks = list(range(0, n, max(1, n // 8)))
            ax_r.set_xticks(ticks)
            labels = []
            for i in ticks:
                dt = dp["date"].iloc[i]
                if hasattr(dt, 'astimezone'):
                    dt_et = dt.astimezone(EASTERN)
                    labels.append(dt_et.strftime("%H:%M"))
                else:
                    labels.append(dt.strftime("%H:%M"))
            ax_r.set_xticklabels(labels, rotation=45, fontsize=7)
        ax_p.set_xticklabels([])
        ax_v.set_xticklabels([])

        sc = COLORS["bull"] if score >= 6 else COLORS["trigger"] if score >= 5 else COLORS["bear"]
        checks_str = " ✓".join([""] + passed_checks)
        vol_warn = " [⚠️ VOL]" if not signal.volume_confirmed else ""
        fig.suptitle(f"{SYMBOL} {direction} Signal — {score}/{total}{vol_warn}{checks_str}",
                     fontsize=13, fontweight="bold", color=sc, y=0.96)

        now_et = datetime.now(EASTERN)
        fig.text(0.99, 0.01, f"OptiPulse Phase 2.1 | {primary} | Signal: {signal.signal_candle_time} ET",
                 fontsize=7, color=COLORS["text"], alpha=0.5, ha="right", va="bottom")

        if contracts_text:
            fig.text(0.02, 0.01, contracts_text, fontsize=7, color=COLORS["text"], alpha=0.8,
                     ha="left", va="bottom", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", fc=COLORS["bg"], ec=COLORS["grid"], alpha=0.9))

        ts = now_et.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CHART_DIR, f"{SYMBOL}_{direction}_{score}of{total}_{ts}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)
        log.info(f"  📸 Chart: {path}")
        return path
    except Exception as e:
        log.error(f"Chart error: {e}")
        plt.close("all")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# DISCORD — PHASE 2.1
# ══════════════════════════════════════════════════════════════════════════════

def _post(content, filepath=None):
    if not WEBHOOK_URL:
        log.warning("⚠️  WEBHOOK_URL not set")
        return
    try:
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                requests.post(WEBHOOK_URL, data={"content": content},
                              files={"file": (os.path.basename(filepath), f, "image/png")}, timeout=15)
        else:
            requests.post(WEBHOOK_URL, json={"content": content}, timeout=10)
    except Exception as e:
        log.error(f"Discord error: {e}")

def send_signal_alert(signal, contracts, chart_path=None, latency_seconds=0, contract_warning=""):
    de = "🟢" if signal.direction == "CALL" else "🔴"
    bar = "🟩" * signal.score + "⬛" * (signal.total - signal.score)
    
    # PHASE 2.1: Volume warning
    vol_warning = ""
    if not signal.volume_confirmed:
        vol_warning = "\n⚠️ **Volume not confirmed** - verify entry manually"

    check_lines = "\n".join(
        f"{'✅' if c.passed else '⚠️' if c.warning else '❌'} {c.name}: {c.detail}" 
        for c in signal.checks
    )
    tf_line = " | ".join(f"{tf}: {d}" for tf, d in signal.tf_details.items())

    # PHASE 2.2: Show contract error if no contracts
    if contracts:
        contract_lines = "\n".join(f"  #{i+1} {p.line()}" for i, p in enumerate(contracts))
    else:
        contract_lines = "  ⚠️ None found - see error below"

    entry_stop = f"\n**Entry:** ${signal.entry_price:.2f} | **Stop:** ${signal.stop_loss:.2f}"
    
    alert_time_et = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
    timing = f"\n**Signal Candle:** {signal.signal_candle_time} ET\n**Alert Sent:** {alert_time_et} ET\n**Latency:** {latency_seconds}s"

    msg = (f"{de} **{SYMBOL} {signal.direction} Signal — {signal.score}/{signal.total}**\n"
           f"{bar}{vol_warning}\n\n**Checks:**\n{check_lines}\n\n"
           f"**Timeframes:** {tf_line}\n"
           f"{entry_stop}\n"
           f"**Top Contracts:**\n{contract_lines}"
           f"{contract_warning}\n"
           f"{timing}")

    if len(msg) > 1900:
        msg = msg[:1900] + "..."
    _post(msg, chart_path)

def send_startup():
    now_et = datetime.now(EASTERN)
    _post(f"🚀 **OptiPulse Phase 2.1 Started**\n"
          f"Symbol: {SYMBOL}\nTimeframes: {', '.join(TIMEFRAMES.keys())}\n"
          f"Score: {MIN_SCORE}/{TOTAL_CHECKS}\nDelta: {DELTA_MIN}-{DELTA_MAX}\n"
          f"DTE Range: {MIN_DTE}-{MAX_DTE} days\n"
          f"Multi-TF: 5m req, 15m not-opposite OK\n"
          f"Volume: Warning only (not blocking)\n"
          f"Daily Reset: ✅ at 9:30 ET\n"
          f"Interval: {SCAN_INTERVAL}s\n⏰ {now_et:%I:%M:%S %p ET}")

def send_shutdown():
    now_et = datetime.now(EASTERN)
    _post(f"⏹️ **OptiPulse Phase 2.1 Stopped** — {now_et:%I:%M:%S %p ET}")

def send_eod_summary(total_scans, total_signals, signals_log, shadow_results):
    sig_lines = "\n".join(
        f"  {s['time']} — {s['direction']} {s['score']}/{s['total']} ${s.get('strike','N/A')}"
        for s in signals_log[-10:]
    ) or "  No signals today"

    shadow_lines = ""
    if shadow_results:
        w, l = shadow_results.get("wins", 0), shadow_results.get("losses", 0)
        t = w + l
        wr = (w / t * 100) if t else 0
        shadow_lines = f"  Trades: {t} | W/L: {w}/{l} | Win Rate: {wr:.0f}%"
        if "avg_pnl" in shadow_results:
            shadow_lines += f"\n  Avg P&L: ${shadow_results['avg_pnl']:.2f}"
    else:
        shadow_lines = "  No shadow data yet"

    now_et = datetime.now(EASTERN)
    _post(f"📊 **EOD Summary — {now_et:%Y-%m-%d}**\n\n"
          f"**Stats:** Scans: {total_scans} | Signals: {total_signals}\n\n"
          f"**Recent Signals:**\n{sig_lines}\n\n"
          f"**Shadow Test:**\n{shadow_lines}\n\n"
          f"⏰ {now_et:%I:%M:%S %p ET}")

# ══════════════════════════════════════════════════════════════════════════════
# SHADOW TRACKER (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShadowTrade:
    timestamp: str
    symbol: str
    direction: str
    score: int
    total: int
    passed_checks: List[str]
    entry_price: float
    strike: float
    expiry: str
    delta: float
    iv: float
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "OPEN"

class ShadowTracker:
    def __init__(self):
        self.trades: List[ShadowTrade] = []
        self.start_date = datetime.now().strftime("%Y-%m-%d")
        self._load()

    def _path(self):
        return os.path.join(SHADOW_LOG_DIR, f"shadow_{self.start_date}.json")

    def _load(self):
        p = self._path()
        if os.path.exists(p):
            try:
                with open(p) as f:
                    self.trades = [ShadowTrade(**t) for t in json.load(f)]
            except:
                self.trades = []

    def _save(self):
        try:
            with open(self._path(), "w") as f:
                json.dump([asdict(t) for t in self.trades], f, indent=2)
        except Exception as e:
            log.error(f"Shadow save error: {e}")

    def record_entry(self, direction, score, total, passed_checks, entry_price,
                     strike, expiry, delta, iv):
        trade = ShadowTrade(
            timestamp=datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S"),
            symbol=SYMBOL, direction=direction, score=score, total=total,
            passed_checks=passed_checks, entry_price=entry_price,
            strike=strike, expiry=expiry, delta=delta, iv=iv,
        )
        self.trades.append(trade)
        self._save()
        log.info(f"  📝 Shadow: {direction} ${strike} @ ${entry_price:.2f}")

    def check_exits(self, prices):
        for t in self.trades:
            if t.status != "OPEN":
                continue
            key = f"{t.strike}_{t.expiry}_{t.direction[0]}"
            if key in prices:
                cp = prices[key]
                pnl = cp - t.entry_price
                pnl_pct = (pnl / t.entry_price * 100) if t.entry_price > 0 else 0
                if pnl_pct >= 20:
                    t.exit_price, t.pnl, t.pnl_pct, t.status = cp, pnl, pnl_pct, "WIN"
                    t.exit_time = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
                elif pnl_pct <= -30:
                    t.exit_price, t.pnl, t.pnl_pct, t.status = cp, pnl, pnl_pct, "LOSS"
                    t.exit_time = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def expire_old(self):
        cutoff = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        for t in self.trades:
            if t.status == "OPEN" and t.timestamp < cutoff:
                t.status = "EXPIRED"
                t.exit_time = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def get_cumulative(self):
        all_trades = []
        for f in sorted(os.listdir(SHADOW_LOG_DIR)):
            if f.startswith("shadow_") and f.endswith(".json"):
                try:
                    with open(os.path.join(SHADOW_LOG_DIR, f)) as fh:
                        all_trades.extend(json.load(fh))
                except:
                    pass
        w = sum(1 for t in all_trades if t.get("status") == "WIN")
        l = sum(1 for t in all_trades if t.get("status") == "LOSS")
        pnls = [t["pnl"] for t in all_trades if t.get("pnl") is not None]
        return {"wins": w, "losses": l, "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
                "days": len(set(t["timestamp"][:10] for t in all_trades))}

    def get_signals_log(self):
        return [{"time": t.timestamp[11:19], "direction": t.direction,
                 "score": t.score, "total": t.total, "strike": t.strike,
                 "status": t.status} for t in self.trades]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE — PHASE 2.1 TUNED
# ══════════════════════════════════════════════════════════════════════════════

class OptiPulseEngine:
    def __init__(self):
        self.ib = IB()
        self.stock = None
        self.shadow = ShadowTracker()
        self.total_scans = 0
        self.total_signals = 0
        
        # PHASE 2.1: Alert state tracking
        self.last_alert_time = {}
        self.last_signal_state = {}
        self._daily_reset_done = False
        
        self.eod_sent = False

    def connect(self):
        log.info(f"Connecting to IBKR {IB_HOST}:{IB_PORT}...")
        self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=60, readonly=True)
        self.ib.reqMarketDataType(3)
        self.stock = Stock(SYMBOL, "SMART", "USD")
        self.ib.qualifyContracts(self.stock)
        log.info(f"✅ Connected — {SYMBOL}")

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()

    def is_market_hours(self):
        now = datetime.now(EASTERN)
        o = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
        c = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0, microsecond=0)
        
        # PHASE 2.1: Daily reset at market open
        if DAILY_RESET_AT_OPEN and now.hour == 9 and 30 <= now.minute < 32 and not self._daily_reset_done:
            log.info("📅 Market open - resetting alert state for fresh start")
            self.last_alert_time = {}
            self.last_signal_state = {}
            self._daily_reset_done = True
        elif now.hour != 9 or now.minute < 30:
            self._daily_reset_done = False
        
        return o <= now <= c

    def should_send_alert(self, signal):
        """
        PHASE 2.1: Relaxed alert logic
        - First qualify each day = alert
        - Cooldown per direction
        - Direction flip overrides
        - No "score unchanged" blocking
        """
        direction = signal.direction
        score = signal.score
        now = time.time()
        
        # Check last state
        last_state = self.last_signal_state.get(SYMBOL, {})
        last_direction = last_state.get("direction")
        last_score = last_state.get("score", 0)
        
        # PHASE 2.1: Direction flip override
        if ALLOW_DIRECTION_FLIP and last_direction and last_direction != direction:
            log.info(f"  🔄 Direction flip: {last_direction} → {direction} (SEND)")
            return True, "DIRECTION_FLIP"
        
        # PHASE 2.1: First qualify of the day = always alert
        if not last_direction or last_score < MIN_SCORE:
            log.info(f"  ✨ First qualifying signal today (SEND)")
            return True, "FIRST_QUALIFY"
        
        # PHASE 2.1: Cooldown check (but less strict)
        last_alert = self.last_alert_time.get(direction, 0)
        time_since = now - last_alert
        
        if time_since < ALERT_COOLDOWN_SECONDS:
            remaining = int(ALERT_COOLDOWN_SECONDS - time_since)
            log.info(f"  ⏳ Cooldown: {remaining}s remaining for {direction} (SKIP: COOLDOWN)")
            return False, "COOLDOWN"
        
        # PHASE 2.1: If cooldown passed and still qualifies = alert
        log.info(f"  ✅ Cooldown passed, signal still valid (SEND)")
        return True, "COOLDOWN_PASSED"

    def scan(self):
        self.total_scans += 1
        log.info(f"═══ Scan #{self.total_scans} ═══")
        
        scan_start = time.time()

        try:
            price = get_current_price(self.ib, self.stock)
            log.info(f"💰 {SYMBOL}: ${price:.2f}")

            log.info("📊 Collecting bars...")
            tf_data = collect_all_timeframes(self.ib, self.stock)

            active = sum(1 for v in tf_data.values() if v is not None)
            if active < 2:
                log.warning(f"⚠️  Only {active} TFs, need 2+ (SKIP: INSUFFICIENT_DATA)")
                return

            log.info("🧮 Scoring...")
            signal = score_multi_timeframe(tf_data)
            log.info(f"  → {signal.summary()}")
            
            for c in signal.checks:
                emoji = '✅' if c.passed else ('⚠️' if c.warning else '❌')
                log.info(f"    {emoji} {c.name}: {c.detail}")

            # Update state
            self.last_signal_state[SYMBOL] = {
                "direction": signal.direction,
                "score": signal.score
            }

            if not signal.qualifies:
                log.info(f"  {signal.score}/{signal.total} < {MIN_SCORE} — skip (SKIP: SCORE_TOO_LOW)")
                return

            # PHASE 2.1: Check if we should alert
            should_alert, reason = self.should_send_alert(signal)
            if not should_alert:
                log.info(f"  Skip reason: {reason}")
                return

            log.info(f"🚨 SIGNAL: {signal.direction} {signal.score}/{signal.total} (Reason: {reason})")

            log.info("🎯 Picking contracts...")
            contracts, contract_error = pick_top_contracts(self.ib, self.stock, price, signal.direction)
            
            # PHASE 2.2: Always alert, even if no contracts
            # Client wants to see signal and manually check Webull
            contract_warning = ""
            if not contracts:
                log.warning(f"⚠️  No tradable contracts: {contract_error}")
                contract_warning = f"\n\n⚠️ **CONTRACTS: {contract_error}**\nManually check Webull for tradable options."

            log.info("📸 Charting...")
            if contracts:
                ct = "\n".join(f"#{i+1} ${p.strike} {p.display_right()} Δ{p.delta:.2f} ${p.price:.2f}"
                               for i, p in enumerate(contracts))
            else:
                ct = f"⚠️ {contract_error or 'No contracts found'}"
            
            chart_path = generate_chart(tf_data, signal, ct)

            latency = int(time.time() - scan_start)

            log.info(f"📤 Sending alert... (Reason: {reason})")
            send_signal_alert(signal, contracts, chart_path, latency, contract_warning)

            if contracts:
                top = contracts[0]
                self.shadow.record_entry(
                    signal.direction, signal.score, signal.total, signal.passed_names,
                    top.price, top.strike, top.expiry, top.delta, top.iv,
                )

            self.last_alert_time[signal.direction] = time.time()
            self.total_signals += 1

        except Exception as e:
            log.error(f"Scan error: {e}", exc_info=True)

    def send_eod(self):
        if self.eod_sent:
            return
        log.info("📊 Sending EOD...")
        self.shadow.expire_old()
        cum = self.shadow.get_cumulative()
        send_eod_summary(self.total_scans, self.total_signals,
                         self.shadow.get_signals_log(), cum)
        self.eod_sent = True

    def run(self):
        log.info(f"🚀 OptiPulse Phase 2.1 — {SYMBOL}")
        log.info(f"  TFs: {list(TIMEFRAMES.keys())} | Score: {MIN_SCORE}/{TOTAL_CHECKS}")
        log.info(f"  Multi-TF: 5m req, 15m not-opposite | Volume: Warning only")
        log.info(f"  Daily Reset: {DAILY_RESET_AT_OPEN} at 9:30 ET")
        log.info(f"  DTE: {MIN_DTE}-{MAX_DTE} | Δ: {DELTA_MIN}-{DELTA_MAX} | Spread: <{MAX_SPREAD_PERCENT*100:.0f}%")
        log.info(f"  Cooldown: {ALERT_COOLDOWN_SECONDS}s | Interval: {SCAN_INTERVAL}s\n")

        self.connect()
        send_startup()

        try:
            while True:
                now = datetime.now(EASTERN)

                if not self.is_market_hours():
                    if now.hour == 16 and now.minute < 15:
                        self.send_eod()
                    log.info("🌙 Market closed — 60s")
                    time.sleep(60)
                    if now.hour == 0 and now.minute < 2:
                        self.eod_sent = False
                        self.total_scans = self.total_signals = 0
                        self.shadow = ShadowTracker()
                        self.last_alert_time = {}
                        self.last_signal_state = {}
                        self._daily_reset_done = False
                    continue

                self.eod_sent = False

                if not self.ib.isConnected():
                    log.warning("🔄 Reconnecting...")
                    try:
                        self.connect()
                    except Exception as e:
                        log.error(f"Reconnect failed: {e}")
                        time.sleep(30)
                        continue

                self.scan()
                log.info(f"⏱️  Next in {SCAN_INTERVAL}s\n")
                time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            log.info("\n👋 Shutting down...")
            self.send_eod()
            send_shutdown()
        except Exception as e:
            log.error(f"Fatal: {e}", exc_info=True)
            _post(f"⚠️ Fatal error: {e}")
        finally:
            self.disconnect()


if __name__ == "__main__":
    OptiPulseEngine().run()