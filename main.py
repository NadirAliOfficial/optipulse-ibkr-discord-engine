

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

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
# CONFIG
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

MIN_SCORE = 5
TOTAL_CHECKS = 7

DELTA_MIN, DELTA_MAX = 0.30, 0.60
STRIKE_RANGE = 0.10
MAX_EXPIRATIONS = 2
BATCH_SIZE = 40

SCAN_INTERVAL = 60
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 0, 0
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 23, 59
ALERT_COOLDOWN = 900

SHADOW_LOG_DIR = "shadow_logs"
CHART_DIR = "charts"

os.makedirs(SHADOW_LOG_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("optipulse")
logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
logging.getLogger("ib_insync.client").setLevel(logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
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
    """Add all indicator columns to a DataFrame."""
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
# BAR COLLECTOR
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
# SCORING ENGINE — 7 CHECKS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

@dataclass
class SignalScore:
    direction: str
    score: int
    total: int
    checks: List[CheckResult]
    tf_details: Dict[str, str] = field(default_factory=dict)

    @property
    def passed_names(self):
        return [c.name for c in self.checks if c.passed]

    @property
    def qualifies(self):
        return self.score >= MIN_SCORE

    def summary(self):
        return f"{self.direction} {self.score}/{self.total} [{' | '.join(self.passed_names)}]"

def _score_single_tf(df):
    last = df.iloc[-1]
    bullish = last["ema_fast"] > last["ema_slow"]
    direction = "CALL" if bullish else "PUT"
    checks = {}

    # EMA
    checks["ema"] = {"passed": True, "detail": f"EMA9={last['ema_fast']:.2f} vs EMA21={last['ema_slow']:.2f}"}

    # VWAP
    if pd.notna(last["vwap"]):
        vwap_ok = last["close"] > last["vwap"] if bullish else last["close"] < last["vwap"]
        checks["vwap"] = {"passed": vwap_ok, "detail": f"Close={last['close']:.2f} vs VWAP={last['vwap']:.2f}"}
    else:
        checks["vwap"] = {"passed": False, "detail": "VWAP N/A"}

    # RSI
    rsi = last["rsi"]
    if bullish:
        rsi_ok = RSI_CALL_MIN <= rsi <= RSI_CALL_MAX
    else:
        rsi_ok = RSI_PUT_MIN <= rsi <= RSI_PUT_MAX
    checks["rsi"] = {"passed": rsi_ok, "detail": f"RSI={rsi:.1f}"}

    # Volume
    checks["volume"] = {"passed": bool(last["vol_expansion"]), "detail": f"Vol={last['volume']:.0f}"}

    # ATR
    atr_ok = last["candle_range"] > 0.5 * last["atr"] if pd.notna(last["atr"]) else False
    checks["atr"] = {"passed": atr_ok,
                      "detail": f"Range={last['candle_range']:.2f} vs 0.5*ATR={0.5 * last['atr']:.2f}" if pd.notna(last["atr"]) else "N/A"}

    # Chop
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

    primary = tf_scores.get("5m") or list(tf_scores.values())[0]
    direction = primary["direction"]
    checks = []

    # 1-5: from primary TF
    for name, label in [("ema", "EMA"), ("vwap", "VWAP"), ("rsi", "RSI"),
                         ("volume", "Volume"), ("atr", "ATR")]:
        c = primary["checks"][name]
        checks.append(CheckResult(label, c["passed"], c["detail"]))

    # 6: Multi-TF agreement
    dirs = [s["direction"] for s in tf_scores.values()]
    agreement = dirs.count(direction)
    checks.append(CheckResult("Multi-TF", agreement >= 2, f"{agreement}/{len(dirs)} TFs agree"))

    # 7: Chop filter (any TF trending)
    chop_pass = any(s["checks"]["chop"]["passed"] for s in tf_scores.values())
    chop_detail = " | ".join(f"{tf}:{s['checks']['chop']['detail']}" for tf, s in tf_scores.items())
    checks.append(CheckResult("Chop", chop_pass, chop_detail))

    score = sum(1 for c in checks if c.passed)
    return SignalScore(direction, score, TOTAL_CHECKS, checks, tf_details)

# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT PICKER — TOP 3
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
    rank_score: float

    def display_right(self):
        return "CALL" if self.right == "C" else "PUT"

    def line(self):
        return (f"${self.strike} {self.display_right()} {self.expiry} | "
                f"${self.price:.2f} Δ{self.delta:.2f} IV:{self.iv:.0%} | "
                f"Spread:${self.spread:.2f}({self.spread_pct:.1%})")

def pick_top_contracts(ib, stock, price, direction, max_picks=3):
    right = "C" if direction == "CALL" else "P"
    chains = ib.reqSecDefOptParams(stock.symbol, "", "STK", stock.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), None)
    if not chain:
        return []

    today = datetime.now().strftime("%Y%m%d")
    expiries = [e for e in sorted(chain.expirations) if e > today][:MAX_EXPIRATIONS]
    strikes = sorted([s for s in chain.strikes
                      if price * (1 - STRIKE_RANGE) <= s <= price * (1 + STRIKE_RANGE)])
    if not strikes:
        return []

    contracts = [Option(stock.symbol, exp, s, right, "SMART") for exp in expiries for s in strikes]
    log.info(f"  Qualifying {len(contracts)} {direction} contracts...")
    ib.qualifyContracts(*contracts)
    valid = [c for c in contracts if c.conId > 0]
    if not valid:
        return []

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
            vol = int(t.volume) if t.volume and t.volume >= 0 else 0
            oi_val = t.callOpenInterest if right == "C" else t.putOpenInterest
            oi = int(oi_val) if oi_val and oi_val >= 0 else 0

            # Rank: delta proximity 30%, spread 35%, liquidity 35%
            d_score = abs(delta - 0.45) / 0.15
            s_score = min(spread_pct / 0.10, 1.0)
            l_score = max(0, 1.0 - ((vol + oi) / 500))
            rank = 0.30 * d_score + 0.35 * s_score + 0.35 * l_score

            picks.append(ContractPick(
                t.contract.lastTradeDateOrContractMonth, t.contract.strike,
                right, price_opt, bid, ask, spread, spread_pct,
                delta, iv, vol, oi, rank,
            ))

        for t in tickers:
            try: ib.cancelMktData(t.contract)
            except: pass

    picks.sort(key=lambda p: p.rank_score)
    top = picks[:max_picks]
    for i, p in enumerate(top):
        log.info(f"  #{i+1}: {p.line()}")
    return top

# ══════════════════════════════════════════════════════════════════════════════
# CHARTING — ANNOTATED SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#1a1a2e", "panel": "#16213e", "text": "#e0e0e0", "grid": "#2a2a4a",
    "bull": "#00e676", "bear": "#ff1744", "ema_fast": "#ffab00", "ema_slow": "#448aff",
    "vwap": "#e040fb", "rsi": "#26c6da", "vol_bar": "#546e7a", "vol_hi": "#ffab00",
    "trigger": "#ffd600",
}

def generate_chart(tf_data, direction, score, total, passed_checks, contracts_text=""):
    primary = None
    for tf in ["5m", "1m", "15m"]:
        if tf in tf_data and tf_data[tf] is not None:
            primary = tf
            break
    if not primary:
        return None

    df = tf_data[primary].copy()
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

        # Candlesticks
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

        # Trigger candle
        ti = n - 1
        ax_p.axvline(x=ti, color=COLORS["trigger"], lw=2, alpha=0.4, ls="--")
        arrow_emoji = "▲" if direction == "CALL" else "▼"
        ax_p.annotate(f"{arrow_emoji} TRIGGER", xy=(ti, dp.iloc[-1]["high"] * 1.002),
                       fontsize=9, fontweight="bold", color=COLORS["trigger"], ha="center", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg"], ec=COLORS["trigger"], alpha=0.8))

        ax_p.legend(loc="upper left", fontsize=8, facecolor=COLORS["panel"],
                     edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
        ax_p.set_ylabel("Price", color=COLORS["text"], fontsize=9)

        # Volume
        vc = [COLORS["vol_hi"] if dp.iloc[i].get("vol_expansion", False) else COLORS["vol_bar"] for i in range(n)]
        ax_v.bar(x, dp["volume"].values, color=vc, alpha=0.7, width=0.7)
        ax_v.set_ylabel("Vol", color=COLORS["text"], fontsize=9)

        # RSI
        ax_r.plot(x, dp["rsi"].values, color=COLORS["rsi"], lw=1.2)
        ax_r.axhline(70, color=COLORS["bear"], lw=0.7, alpha=0.5, ls="--")
        ax_r.axhline(30, color=COLORS["bull"], lw=0.7, alpha=0.5, ls="--")
        ax_r.axhline(50, color=COLORS["text"], lw=0.5, alpha=0.3, ls=":")
        ax_r.set_ylim(10, 90)
        ax_r.set_ylabel("RSI", color=COLORS["text"], fontsize=9)

        # X labels
        if "date" in dp.columns:
            ticks = list(range(0, n, max(1, n // 8)))
            ax_r.set_xticks(ticks)
            ax_r.set_xticklabels([dp["date"].iloc[i].strftime("%H:%M") for i in ticks], rotation=45, fontsize=7)
        ax_p.set_xticklabels([])
        ax_v.set_xticklabels([])

        # Title
        sc = COLORS["bull"] if score >= 6 else COLORS["trigger"] if score >= 5 else COLORS["bear"]
        checks_str = " ✓".join([""] + passed_checks)
        fig.suptitle(f"{SYMBOL} {direction} Signal — {score}/{total}{checks_str}",
                     fontsize=13, fontweight="bold", color=sc, y=0.96)

        fig.text(0.99, 0.01, f"OptiPulse | {primary} | {datetime.now():%Y-%m-%d %H:%M:%S}",
                 fontsize=7, color=COLORS["text"], alpha=0.5, ha="right", va="bottom")

        if contracts_text:
            fig.text(0.02, 0.01, contracts_text, fontsize=7, color=COLORS["text"], alpha=0.8,
                     ha="left", va="bottom", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", fc=COLORS["bg"], ec=COLORS["grid"], alpha=0.9))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
# DISCORD
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

def send_signal_alert(signal, contracts, chart_path=None):
    de = "🟢" if signal.direction == "CALL" else "🔴"
    bar = "🟩" * signal.score + "⬛" * (signal.total - signal.score)

    check_lines = "\n".join(f"{'✅' if c.passed else '❌'} {c.name}: {c.detail}" for c in signal.checks)
    tf_line = " | ".join(f"{tf}: {d}" for tf, d in signal.tf_details.items())

    contract_lines = "\n".join(f"  #{i+1} {p.line()}" for i, p in enumerate(contracts)) if contracts else "  None found"

    msg = (f"{de} **{SYMBOL} {signal.direction} Signal — {signal.score}/{signal.total}**\n"
           f"{bar}\n\n**Checks:**\n{check_lines}\n\n"
           f"**Timeframes:** {tf_line}\n\n"
           f"**Top Contracts:**\n{contract_lines}\n\n"
           f"⏰ {datetime.now():%I:%M:%S %p}")

    if len(msg) > 1900:
        msg = msg[:1900] + "..."
    _post(msg, chart_path)

def send_startup():
    _post(f"🚀 **OptiPulse Phase 2 Started**\n"
          f"Symbol: {SYMBOL}\nTimeframes: {', '.join(TIMEFRAMES.keys())}\n"
          f"Score: {MIN_SCORE}/{TOTAL_CHECKS}\nDelta: {DELTA_MIN}-{DELTA_MAX}\n"
          f"Interval: {SCAN_INTERVAL}s\n⏰ {datetime.now():%I:%M:%S %p}")

def send_shutdown():
    _post(f"⏹️ **OptiPulse Phase 2 Stopped** — {datetime.now():%I:%M:%S %p}")

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

    _post(f"📊 **EOD Summary — {datetime.now():%Y-%m-%d}**\n\n"
          f"**Stats:** Scans: {total_scans} | Signals: {total_signals}\n\n"
          f"**Recent Signals:**\n{sig_lines}\n\n"
          f"**Shadow Test:**\n{shadow_lines}\n\n"
          f"⏰ {datetime.now():%I:%M:%S %p}")

# ══════════════════════════════════════════════════════════════════════════════
# SHADOW TRACKER
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
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                    t.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elif pnl_pct <= -30:
                    t.exit_price, t.pnl, t.pnl_pct, t.status = cp, pnl, pnl_pct, "LOSS"
                    t.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def expire_old(self):
        cutoff = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        for t in self.trades:
            if t.status == "OPEN" and t.timestamp < cutoff:
                t.status = "EXPIRED"
                t.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class OptiPulseEngine:
    def __init__(self):
        self.ib = IB()
        self.stock = None
        self.shadow = ShadowTracker()
        self.total_scans = 0
        self.total_signals = 0
        self.last_alert = {}
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
        now = datetime.now()
        o = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
        c = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
        return o <= now <= c

    def scan(self):
        self.total_scans += 1
        log.info(f"═══ Scan #{self.total_scans} ═══")

        try:
            price = get_current_price(self.ib, self.stock)
            log.info(f"💰 {SYMBOL}: ${price:.2f}")

            log.info("📊 Collecting bars...")
            tf_data = collect_all_timeframes(self.ib, self.stock)

            active = sum(1 for v in tf_data.values() if v is not None)
            if active < 2:
                log.warning(f"⚠️  Only {active} TFs, need 2+")
                return

            log.info("🧮 Scoring...")
            signal = score_multi_timeframe(tf_data)
            log.info(f"  → {signal.summary()}")
            for c in signal.checks:
                log.info(f"    {'✅' if c.passed else '❌'} {c.name}: {c.detail}")

            if not signal.qualifies:
                log.info(f"  {signal.score}/{signal.total} < {MIN_SCORE} — skip")
                return

            key = f"{signal.direction}_{signal.score}"
            if key in self.last_alert and (time.time() - self.last_alert[key]) < ALERT_COOLDOWN:
                log.info(f"  ⏳ Cooldown for {key}")
                return

            log.info(f"🚨 SIGNAL: {signal.direction} {signal.score}/{signal.total}")

            log.info("🎯 Picking contracts...")
            contracts = pick_top_contracts(self.ib, self.stock, price, signal.direction)

            log.info("📸 Charting...")
            ct = "\n".join(f"#{i+1} ${p.strike} {p.display_right()} Δ{p.delta:.2f} ${p.price:.2f}"
                           for i, p in enumerate(contracts)) if contracts else ""
            chart_path = generate_chart(tf_data, signal.direction, signal.score,
                                         signal.total, signal.passed_names, ct)

            log.info("📤 Sending alert...")
            send_signal_alert(signal, contracts, chart_path)

            if contracts:
                top = contracts[0]
                self.shadow.record_entry(
                    signal.direction, signal.score, signal.total, signal.passed_names,
                    top.price, top.strike, top.expiry, top.delta, top.iv,
                )

            self.last_alert[key] = time.time()
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
        log.info(f"🚀 OptiPulse Phase 2 — {SYMBOL}")
        log.info(f"  TFs: {list(TIMEFRAMES.keys())} | Score: {MIN_SCORE}/{TOTAL_CHECKS} | Interval: {SCAN_INTERVAL}s\n")

        self.connect()
        send_startup()

        try:
            while True:
                now = datetime.now()

                if not self.is_market_hours():
                    if now.hour == MARKET_CLOSE_HOUR and now.minute < 15:
                        self.send_eod()
                    log.info("🌙 Market closed — 60s")
                    time.sleep(60)
                    if now.hour == 0 and now.minute < 2:
                        self.eod_sent = False
                        self.total_scans = self.total_signals = 0
                        self.shadow = ShadowTracker()
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