
import os, time, threading, requests, numpy as np, pandas as pd
from collections import defaultdict, deque
from flask import Flask, jsonify

# ============================================================
# CRYPTO OPTIONS TELEGRAM SIGNAL BOT
# ============================================================
# DATA:
#   OKX public API = primary spot + option data
#   Deribit public API = fallback for BTC/ETH options
#
# MANDATORY SIGNAL PARAMETERS:
#   1. VWAP
#   2. EMA 9 / EMA 20
#   3. RSI 14
#   4. ADX 14
#
# TIMEFRAME: 5 MINUTES
# SIGNALS:
#   EARLY CALL / PUT WATCH
#   CONFIRMED CALL BUY / PUT BUY
#
# NO AUTOMATIC ORDERS ARE PLACED.
# ============================================================

BOT_NAME = os.getenv("BOT_NAME", "Crypto 5M VWAP EMA9/20 RSI ADX Options Bot")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8643574412:AAFJXnkpGXuQhMXBCbbwZxsehPHlFmJkO0c").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "8991485495").strip()

OKX_BASE = os.getenv("OKX_BASE", "https://www.okx.com").rstrip("/")
DERIBIT_BASE = os.getenv("DERIBIT_BASE", "https://www.deribit.com/api/v2").rstrip("/")

INTERVAL = "5m"
HISTORY = int(os.getenv("HISTORY", "180"))
SCAN_SLEEP = float(os.getenv("SCAN_SLEEP", "2.0"))

# ---------------- MANDATORY PARAMETERS ----------------
EMA_FAST = 9
EMA_SLOW = 20
RSI_LEN = 14
ADX_LEN = 14
VWAP_REQUIRED = True

# Signal thresholds
MIN_ADX = float(os.getenv("MIN_ADX", "20"))
STRONG_ADX = float(os.getenv("STRONG_ADX", "25"))
RSI_LONG = float(os.getenv("RSI_LONG", "55"))
RSI_SHORT = float(os.getenv("RSI_SHORT", "45"))
EARLY_RSI_LONG = float(os.getenv("EARLY_RSI_LONG", "52"))
EARLY_RSI_SHORT = float(os.getenv("EARLY_RSI_SHORT", "48"))

# Secondary confirmations
VOL_LEN = 20
MIN_VOL_RATIO = float(os.getenv("MIN_VOL_RATIO", "1.00"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
EARLY_MIN_SCORE = int(os.getenv("EARLY_MIN_SCORE", "6"))

# Option filters
MAX_STRIKE_DISTANCE = float(os.getenv("MAX_STRIKE_DISTANCE", "0.04"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "0.30"))
MAX_DELTA = float(os.getenv("MAX_DELTA", "0.70"))
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.05"))
MIN_DTE_HOURS = float(os.getenv("MIN_DTE_HOURS", "12"))

# Premium management
SL_PCT = float(os.getenv("OPTION_SL_PCT", "0.25"))
T1_PCT = float(os.getenv("OPTION_T1_PCT", "0.40"))
T2_PCT = float(os.getenv("OPTION_T2_PCT", "0.70"))

EARLY_COOLDOWN = int(os.getenv("EARLY_COOLDOWN_MIN", "20")) * 60
CONF_COOLDOWN = int(os.getenv("CONF_COOLDOWN_MIN", "10")) * 60
OPTION_REFRESH = int(os.getenv("OPTION_REFRESH_MIN", "20")) * 60

REQUESTED = [
    ("Bitcoin", "BTC"),
    ("Ethereum", "ETH"),
    ("Solana", "SOL"),
    ("Ripple", "XRP"),
    ("Dogecoin", "DOGE"),
    ("Cardano", "ADA"),
    ("Avalanche", "AVAX"),
    ("Tron", "TRX"),
    ("Binance Coin", "BNB"),
    ("Near Protocol", "NEAR"),
    ("Aave", "AAVE"),
    ("Lighter", "LIT"),
    ("Ethena", "ENA"),
    ("Zcash", "ZEC"),
    ("Akedo", "AKE"),
    ("Esports Token", "ESPORTS"),
    ("Uniswap", "UNI"),
    ("Chainlink", "LINK"),
    ("Litecoin", "LTC"),
    ("Polygon", "MATIC"),
]

ALIASES = {
    "MATIC": ["MATIC", "POL"],
}

app = Flask(__name__)
lock = threading.RLock()

spot_symbols = {}          # asset -> OKX SPOT instId
asset_name = {}
bars = {}                  # asset -> deque
last_bar = {}
last_alert = {}
seen = set()
stats = defaultdict(int)

okx_instruments = {}       # asset -> list option instruments
deribit_instruments = {}   # BTC/ETH -> list
option_market = {}
provider_status = {"okx": "unknown", "deribit": "unknown"}
provider_errors = {}
last_option_refresh = 0


# ============================================================
# HTTP / TELEGRAM
# ============================================================

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 CryptoSignalBot/1.0",
    "Accept": "application/json",
})

def telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("TELEGRAM NOT CONFIGURED:\n", text)
        return False
    try:
        r = SESSION.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=15
        )
        if not r.ok:
            print("Telegram error:", r.status_code, r.text[:500])
        return r.ok
    except Exception as e:
        print("Telegram exception:", e)
        return False

def get_json(url, params=None, timeout=15):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "code" in data and str(data.get("code")) not in ("0", ""):
        raise RuntimeError(f"API error {data.get('code')}: {data.get('msg')}")
    return data

def fmt(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "-"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        if abs(x) >= 1:
            return f"{x:,.4f}"
        return f"{x:,.8f}"
    except Exception:
        return "-"


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return out

def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def adx(df, n=14):
    up = df.high.diff()
    down = -df.low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )

    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs()
    ], axis=1).max(axis=1)

    atrx = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dip = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrx
    dim = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrx
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    ax = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return ax, dip, dim

def build_indicators(rows):
    if len(rows) < 65:
        return None

    d = pd.DataFrame(rows).copy()
    d = d.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    # MANDATORY: EMA 9 / EMA 20
    d["ema9"] = d.close.ewm(span=EMA_FAST, adjust=False).mean()
    d["ema20"] = d.close.ewm(span=EMA_SLOW, adjust=False).mean()

    # MANDATORY: RSI 14
    d["rsi"] = rsi(d.close, RSI_LEN)

    # MANDATORY: ADX 14
    d["adx"], d["di_plus"], d["di_minus"] = adx(d, ADX_LEN)

    # ATR used only for quality / pullback distance
    d["atr"] = atr(d, ADX_LEN)

    # MANDATORY: session/day VWAP
    tp = (d.high + d.low + d.close) / 3.0
    dt = pd.to_datetime(d.open_time, unit="ms", utc=True)
    day = dt.dt.date
    vol_cum = d.volume.groupby(day).cumsum()
    pv_cum = (tp * d.volume).groupby(day).cumsum()
    d["vwap"] = pv_cum / vol_cum.replace(0, np.nan)

    # Secondary volume confirmation
    d["vol_avg"] = d.volume.rolling(VOL_LEN).mean().shift(1)
    d["vol_ratio"] = d.volume / d.vol_avg.replace(0, np.nan)

    rng = (d.high - d.low).replace(0, np.nan)
    d["body"] = (d.close - d.open).abs() / rng
    d["ema_gap_atr"] = (d.ema9 - d.ema20).abs() / d.atr.replace(0, np.nan)

    return d


# ============================================================
# SIGNAL SCORING
# ============================================================

def score(c, side, early=False):
    s = 0

    if side == "LONG":
        s += 2 if c.ema9 > c.ema20 else 0
        s += 2 if c.close > c.vwap else 0
        s += 1 if c.rsi >= (EARLY_RSI_LONG if early else RSI_LONG) else 0
        s += 1 if c.adx > MIN_ADX else 0
        s += 1 if c.di_plus > c.di_minus else 0
    else:
        s += 2 if c.ema9 < c.ema20 else 0
        s += 2 if c.close < c.vwap else 0
        s += 1 if c.rsi <= (EARLY_RSI_SHORT if early else RSI_SHORT) else 0
        s += 1 if c.adx > MIN_ADX else 0
        s += 1 if c.di_minus > c.di_plus else 0

    s += 1 if c.vol_ratio >= MIN_VOL_RATIO else 0
    s += 1 if c.body >= 0.50 else 0
    return min(int(s), 10)

def grade(s):
    if s >= 9:
        return "🔥🔥 VERY HIGH"
    if s >= 8:
        return "🔥 HIGH"
    if s >= 7:
        return "🟢 VALID"
    return "🟡 WATCH"


# ============================================================
# OKX SPOT DISCOVERY + CANDLES
# ============================================================

def discover_okx_spot():
    data = get_json(
        f"{OKX_BASE}/api/v5/public/instruments",
        {"instType": "SPOT"}
    )
    rows = data.get("data", []) if isinstance(data, dict) else []

    live = {}
    for x in rows:
        if str(x.get("state", "")).lower() != "live":
            continue
        inst_id = str(x.get("instId", "")).upper()
        if not inst_id.endswith("-USDT"):
            continue
        base = inst_id[:-5]
        live[base] = inst_id

    found = {}
    for name, asset in REQUESTED:
        inst = None
        for candidate in ALIASES.get(asset, [asset]):
            if candidate.upper() in live:
                inst = live[candidate.upper()]
                break
        if inst:
            found[asset] = inst
            asset_name[asset] = name

    with lock:
        spot_symbols.clear()
        spot_symbols.update(found)

    provider_status["okx"] = f"SPOT OK ({len(found)}/{len(REQUESTED)})"
    print("OKX spot assets:", len(found))
    return len(found)

def okx_candles(inst_id, limit=180):
    # OKX returns newest first.
    data = get_json(
        f"{OKX_BASE}/api/v5/market/candles",
        {
            "instId": inst_id,
            "bar": "5m",
            "limit": str(min(limit, 300))
        }
    )
    rows = data.get("data", [])
    now_ms = int(time.time() * 1000)
    out = []

    for k in reversed(rows):
        # [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]
        if len(k) < 5:
            continue
        ts = int(k[0])
        # confirm == 1 means closed; fallback to timestamp check
        confirmed = len(k) > 8 and str(k[8]) == "1"
        if not confirmed and ts + 5 * 60 * 1000 > now_ms:
            continue
        try:
            out.append({
                "open_time": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]) if len(k) > 5 else 0.0,
                "close_time": ts + 5 * 60 * 1000 - 1
            })
        except Exception:
            pass
    return out[-HISTORY:]

def load_history(asset):
    symbol = spot_symbols.get(asset)
    if not symbol:
        return 0

    rows = okx_candles(symbol, HISTORY)
    with lock:
        bars[asset] = deque(rows[-HISTORY:], maxlen=HISTORY)
        if rows:
            last_bar[asset] = rows[-1]["open_time"]
    return len(rows)

def refresh_asset(asset):
    symbol = spot_symbols.get(asset)
    if not symbol:
        return False

    rows = okx_candles(symbol, 3)
    changed = False

    with lock:
        if asset not in bars:
            bars[asset] = deque(maxlen=HISTORY)

        for b in rows:
            if b["open_time"] > last_bar.get(asset, -1):
                bars[asset].append(b)
                last_bar[asset] = b["open_time"]
                changed = True

    return changed


# ============================================================
# OKX OPTIONS DISCOVERY
# ============================================================

def discover_okx_options():
    data = get_json(
        f"{OKX_BASE}/api/v5/public/instruments",
        {"instType": "OPTION"}
    )
    rows = data.get("data", []) if isinstance(data, dict) else []

    grouped = defaultdict(list)
    for x in rows:
        if str(x.get("state", "")).lower() != "live":
            continue

        inst_id = str(x.get("instId", "")).upper()
        uly = str(x.get("uly", "")).upper()

        # OKX option underlying can be BTC-USD etc.
        base = uly.split("-")[0] if uly else ""
        if not base:
            parts = inst_id.split("-")
            base = parts[0] if parts else ""

        if base in {a for _, a in REQUESTED} or base in {"POL"}:
            grouped[base].append(x)

    with lock:
        okx_instruments.clear()
        for name, asset in REQUESTED:
            candidates = [asset]
            candidates += ALIASES.get(asset, [])
            found = []
            for c in candidates:
                if grouped.get(c):
                    found = grouped[c]
                    break
            if found:
                okx_instruments[asset] = found

    provider_status["okx"] = (
        f"SPOT {len(spot_symbols)} | OPTIONS {len(okx_instruments)}"
    )
    print("OKX option assets:", len(okx_instruments))
    return len(okx_instruments)


# ============================================================
# DERIBIT OPTIONS FALLBACK
# ============================================================

def discover_deribit_options():
    found = {}
    for currency in ("BTC", "ETH"):
        try:
            data = get_json(
                f"{DERIBIT_BASE}/public/get_instruments",
                {
                    "currency": currency,
                    "kind": "option",
                    "expired": "false"
                }
            )
            rows = data.get("result", []) if isinstance(data, dict) else []
            found[currency] = [
                x for x in rows if x.get("is_active", True)
            ]
        except Exception as e:
            provider_errors["deribit"] = str(e)[:300]
            found[currency] = []

    with lock:
        deribit_instruments.clear()
        deribit_instruments.update(found)

    provider_status["deribit"] = (
        f"BTC {len(found.get('BTC', []))} | ETH {len(found.get('ETH', []))}"
    )
    print("Deribit:", provider_status["deribit"])
    return found

def deribit_summaries(currency):
    data = get_json(
        f"{DERIBIT_BASE}/public/get_book_summary_by_currency",
        {"currency": currency, "kind": "option"}
    )
    return data.get("result", []) if isinstance(data, dict) else []


# ============================================================
# OPTION DATA NORMALIZATION
# ============================================================

def parse_expiry_ms(x):
    for key in ("expTime", "expiration_timestamp", "expiryDate", "expiration"):
        if key in x and x.get(key):
            try:
                val = float(x[key])
                if val < 100000000000:
                    val *= 1000
                return int(val)
            except Exception:
                pass
    return 0

def parse_strike(x):
    for key in ("stk", "strike", "strikePrice"):
        try:
            return float(x.get(key))
        except Exception:
            pass
    return 0.0

def parse_option_type(x):
    raw = str(
        x.get("optType") or x.get("option_type") or x.get("side") or ""
    ).upper()
    if raw in ("C", "CALL"):
        return "CALL"
    if raw in ("P", "PUT"):
        return "PUT"
    return ""

def parse_delta(x):
    for key in ("deltaBS", "delta", "greeks_delta"):
        try:
            return float(x.get(key))
        except Exception:
            pass
    greeks = x.get("greeks")
    if isinstance(greeks, dict):
        try:
            return float(greeks.get("delta"))
        except Exception:
            pass
    return np.nan

def normalize_okx_options(asset, spot):
    rows = okx_instruments.get(asset, [])
    if not rows:
        return []

    # OKX option market endpoint gives option summaries/Greeks.
    # Fetch the family/underlying summaries in one call.
    families = set()
    for x in rows:
        uly = str(x.get("uly", "")).upper()
        if uly:
            families.add(uly)

    market_rows = []
    for uly in list(families)[:5]:
        try:
            data = get_json(
                f"{OKX_BASE}/api/v5/public/opt-summary",
                {"uly": uly}
            )
            market_rows.extend(data.get("data", []))
        except Exception as e:
            provider_errors[f"okx_opt_{asset}"] = str(e)[:300]

    by_inst = {
        str(x.get("instId", "")).upper(): x for x in market_rows
    }

    out = []
    now = int(time.time() * 1000)

    for ins in rows:
        inst_id = str(ins.get("instId", "")).upper()
        m = by_inst.get(inst_id, {})

        exp = parse_expiry_ms(ins)
        if exp <= now + MIN_DTE_HOURS * 3600 * 1000:
            continue

        strike = parse_strike(ins)
        if strike <= 0 or spot <= 0:
            continue

        opt_type = parse_option_type(ins)
        if not opt_type:
            continue

        # OKX summary fields vary slightly by version; accept common names.
        def f(keys, default=0.0):
            for k in keys:
                try:
                    v = float(m.get(k))
                    if np.isfinite(v):
                        return v
                except Exception:
                    pass
            return default

        bid = f(["bidPx", "bidPrice"])
        ask = f(["askPx", "askPrice"])
        last = f(["last", "lastPx", "lastPrice"])
        vol = f(["vol24h", "vol24hCcy", "volume"])
        oi = f(["oi", "openInterest"])

        delta = parse_delta(m)
        gamma = f(["gammaBS", "gamma"], np.nan)
        theta = f(["thetaBS", "theta"], np.nan)
        vega = f(["vegaBS", "vega"], np.nan)
        iv = f(["markVol", "markIV", "iv"], np.nan)

        spread = 999.0
        if bid > 0 and ask > 0 and (bid + ask) > 0:
            spread = (ask - bid) / ((ask + bid) / 2)

        out.append({
            "provider": "OKX",
            "asset": asset,
            "symbol": inst_id,
            "expiry_ms": exp,
            "strike": strike,
            "option_type": opt_type,
            "ltp": last,
            "bid": bid,
            "ask": ask,
            "volume": vol,
            "oi": oi,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": iv,
            "spread": spread,
        })

    return out

def normalize_deribit_options(asset, spot):
    if asset not in ("BTC", "ETH"):
        return []

    try:
        summaries = deribit_summaries(asset)
    except Exception as e:
        provider_errors[f"deribit_opt_{asset}"] = str(e)[:300]
        return []

    out = []
    now = int(time.time() * 1000)

    for m in summaries:
        symbol = str(m.get("instrument_name", "")).upper()
        ins = next(
            (x for x in deribit_instruments.get(asset, [])
             if str(x.get("instrument_name", "")).upper() == symbol),
            None
        )
        if not ins:
            continue

        exp = parse_expiry_ms(ins)
        strike = parse_strike(ins)
        opt_type = parse_option_type(ins)

        if exp <= now + MIN_DTE_HOURS * 3600 * 1000:
            continue
        if strike <= 0 or not opt_type:
            continue

        bid = float(m.get("bid_price") or 0)
        ask = float(m.get("ask_price") or 0)
        mark = float(m.get("mark_price") or 0)
        last = float(m.get("last") or mark or 0)
        volume = float(m.get("volume") or 0)
        oi = float(m.get("open_interest") or 0)

        # Deribit option prices are coin-denominated, so convert to USD-like
        # premium only when mark_price is usable with underlying index.
        # Keep the native option premium for the displayed entry.
        delta = float(m.get("delta") or np.nan)
        gamma = float(m.get("gamma") or np.nan)
        theta = float(m.get("theta") or np.nan)
        vega = float(m.get("vega") or np.nan)
        iv = float(m.get("mark_iv") or np.nan)

        spread = 999.0
        if bid > 0 and ask > 0:
            spread = (ask - bid) / ((ask + bid) / 2)

        out.append({
            "provider": "Deribit",
            "asset": asset,
            "symbol": symbol,
            "expiry_ms": exp,
            "strike": strike,
            "option_type": opt_type,
            "ltp": last,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "oi": oi,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": iv,
            "spread": spread,
        })

    return out


# ============================================================
# OPTION SELECTION
# ============================================================

def refresh_option_market():
    global last_option_refresh, option_market

    now = time.time()
    if now - last_option_refresh < OPTION_REFRESH:
        return
    last_option_refresh = now

    combined = {}

    for asset in list(spot_symbols.keys()):
        spot = None
        with lock:
            rows = list(bars.get(asset, []))
        if rows:
            spot = rows[-1]["close"]
        if not spot:
            continue

        try:
            opts = normalize_okx_options(asset, spot)
            if opts:
                combined[asset] = opts
        except Exception as e:
            provider_errors[f"okx_{asset}"] = str(e)[:300]

        # Deribit fallback only if OKX has no options for this asset.
        if asset in ("BTC", "ETH") and not combined.get(asset):
            try:
                opts = normalize_deribit_options(asset, spot)
                if opts:
                    combined[asset] = opts
            except Exception as e:
                provider_errors[f"deribit_{asset}"] = str(e)[:300]

    with lock:
        option_market = combined

    print("Option market refreshed:", {k: len(v) for k, v in combined.items()})


def select_option(asset, direction, spot):
    opts = option_market.get(asset, [])
    if not opts or not spot:
        return None, "NO_OPTION_DATA"

    needed = "CALL" if direction == "LONG" else "PUT"

    # First choose nearest valid expiry with liquidity.
    valid = []
    now = int(time.time() * 1000)

    for o in opts:
        if o["option_type"] != needed:
            continue
        if o["expiry_ms"] <= now + MIN_DTE_HOURS * 3600 * 1000:
            continue
        if o["ltp"] <= 0:
            continue

        dist = abs(o["strike"] - spot) / spot
        if dist > MAX_STRIKE_DISTANCE:
            continue

        if o["spread"] > MAX_SPREAD:
            continue

        delta = o["delta"]
        if np.isfinite(delta):
            ad = abs(delta)
            if ad < MIN_DELTA or ad > MAX_DELTA:
                continue
            delta_score = 1 - abs(ad - 0.50) / 0.50
        else:
            delta_score = 0.40

        volume = max(o["volume"], 0)
        oi = max(o["oi"], 0)

        expiry_score = 1.0 / max((o["expiry_ms"] - now) / 86400000, 0.5)
        distance_score = max(0, 1 - dist / MAX_STRIKE_DISTANCE)
        spread_score = max(0, 1 - o["spread"] / MAX_SPREAD)
        liquidity_score = min(1, np.log1p(volume + oi) / 10)

        selection_score = (
            delta_score * 35
            + distance_score * 30
            + spread_score * 20
            + liquidity_score * 10
            + min(expiry_score, 5)
        )

        row = dict(o)
        row["distance"] = dist
        row["selection_score"] = selection_score
        valid.append(row)

    if not valid:
        # Relax ONLY the liquidity/greeks requirement, not direction or
        # near-ATM/expiry. This prevents "no signals" when an exchange
        # temporarily reports missing Greeks.
        fallback = []
        for o in opts:
            if o["option_type"] != needed or o["ltp"] <= 0:
                continue
            if o["expiry_ms"] <= now + MIN_DTE_HOURS * 3600 * 1000:
                continue
            dist = abs(o["strike"] - spot) / spot
            if dist > MAX_STRIKE_DISTANCE:
                continue
            if o["spread"] > MAX_SPREAD:
                continue
            o2 = dict(o)
            o2["distance"] = dist
            o2["selection_score"] = 100 - dist * 100
            fallback.append(o2)

        if not fallback:
            return None, "NO_LIQUID_NEAR_ATM_OPTION"
        fallback.sort(key=lambda x: x["selection_score"], reverse=True)
        return fallback[0], "FALLBACK_OPTION"

    valid.sort(key=lambda x: x["selection_score"], reverse=True)
    return valid[0], "FILTERED_OPTION"


# ============================================================
# ALERT BUILDERS
# ============================================================

def option_plan(premium):
    return (
        premium * (1 - SL_PCT),
        premium * (1 + T1_PCT),
        premium * (1 + T2_PCT),
    )

def expiry_text(ms):
    try:
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ms / 1000))
    except Exception:
        return "-"

def send_early(asset, sig):
    side_text = "CALL" if sig["side"] == "LONG" else "PUT"
    emoji = "⚡🟢" if sig["side"] == "LONG" else "⚡🔴"

    text = f"""<b>{emoji} EARLY CRYPTO {side_text} WATCH</b>
<b>{asset_name.get(asset, asset)} ({asset})</b>
━━━━━━━━━━━━━━━━━━━━
<b>{grade(sig['score'])}</b> | Score <b>{sig['score']}/10</b>
Timeframe: <b>5 MIN</b>
Trigger: <b>PRE-CROSSOVER MOMENTUM</b>

<b>MANDATORY CONFIRMATIONS</b>
EMA9: <b>{fmt(sig['ema9'])}</b>
EMA20: <b>{fmt(sig['ema20'])}</b>
VWAP: <b>{fmt(sig['vwap'])}</b>
RSI14: <b>{sig['rsi']:.1f}</b>
ADX14: <b>{sig['adx']:.1f}</b>
DI+: {sig['di_plus']:.1f} | DI-: {sig['di_minus']:.1f}

Spot: <b>{fmt(sig['spot'])}</b>
Volume: {sig['vol_ratio']:.2f}x

Expected direction: <b>BUY {side_text}</b>

⚠️ Early signal = watchlist alert.
Wait for EMA9/EMA20 confirmation before option entry."""
    telegram(text)

def send_confirm(asset, sig, option, reason):
    side_text = "CALL" if sig["side"] == "LONG" else "PUT"
    emoji = "🟢" if side_text == "CALL" else "🔴"

    premium = option["ltp"]
    sl, t1, t2 = option_plan(premium)

    text = f"""<b>{emoji} CRYPTO OPTION BUY {side_text}</b>
<b>{asset_name.get(asset, asset)} ({asset})</b>
━━━━━━━━━━━━━━━━━━━━
<b>{grade(sig['score'])}</b> | Score <b>{sig['score']}/10</b>
Timeframe: <b>5 MIN</b>
Trigger: <b>{sig['trigger']}</b>

<b>MANDATORY TECHNICALS</b>
Spot: <b>{fmt(sig['spot'])}</b>
EMA9: <b>{fmt(sig['ema9'])}</b>
EMA20: <b>{fmt(sig['ema20'])}</b>
VWAP: <b>{fmt(sig['vwap'])}</b>
RSI14: <b>{sig['rsi']:.1f}</b>
ADX14: <b>{sig['adx']:.1f}</b>
DI+: {sig['di_plus']:.1f} | DI-: {sig['di_minus']:.1f}
Volume: {sig['vol_ratio']:.2f}x

<b>OPTION</b>
Provider: <b>{option['provider']}</b>
Symbol: <b>{option['symbol']}</b>
Type: <b>{side_text}</b>
Expiry: <b>{expiry_text(option['expiry_ms'])}</b>
Strike: <b>{fmt(option['strike'])}</b>
Premium/LTP: <b>{fmt(premium)}</b>
Bid: {fmt(option.get('bid'))}
Ask: {fmt(option.get('ask'))}
Spread: {option.get('spread', 999)*100:.2f}%
Delta: {fmt(option.get('delta'))}
Gamma: {fmt(option.get('gamma'))}
Theta: {fmt(option.get('theta'))}
Vega: {fmt(option.get('vega'))}
IV: {fmt(option.get('iv'))}
Volume: {fmt(option.get('volume'))}
OI: {fmt(option.get('oi'))}

<b>TRADE PLAN</b>
Entry: <b>{fmt(premium)}</b>
Entry zone: <b>{fmt(premium*0.98)} - {fmt(premium*1.02)}</b>
🛑 SL: <b>{fmt(sl)}</b> (-{SL_PCT*100:.0f}%)
🎯 T1: <b>{fmt(t1)}</b> (+{T1_PCT*100:.0f}%)
🎯 T2: <b>{fmt(t2)}</b> (+{T2_PCT*100:.0f}%)

After T1: move SL to breakeven.
Risk per trade: maximum 0.5-1%.

<b>WHY</b>
✓ VWAP confirmed
✓ EMA9 / EMA20 confirmed
✓ RSI14 confirmed
✓ ADX14 confirmed
✓ DI direction
✓ Volume / candle quality
✓ Near-ATM option
✓ Option liquidity filter

Option selection: <b>{reason}</b>

⚠️ Signal only. No automatic order.
⚠️ Do not chase above the entry zone."""
    telegram(text)


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def make_signal(asset):
    with lock:
        rows = list(bars.get(asset, []))

    d = build_indicators(rows)
    if d is None:
        return None, None, "NOT_ENOUGH_BARS"

    c = d.iloc[-1]
    p = d.iloc[-2]

    required = [
        c.ema9, c.ema20, c.vwap, c.rsi, c.adx,
        c.di_plus, c.di_minus, c.atr, c.vol_ratio
    ]
    if any(pd.isna(x) for x in required):
        return None, None, "INDICATOR_NOT_READY"

    # Mandatory VWAP/EMA/RSI/ADX rules.
    if c.adx <= MIN_ADX:
        return None, None, "ADX_BELOW_MIN"

    cross_long = p.ema9 <= p.ema20 and c.ema9 > c.ema20
    cross_short = p.ema9 >= p.ema20 and c.ema9 < c.ema20

    # Confirmed long/short: ALL FOUR mandatory indicators + DI.
    confirmed_long = (
        c.ema9 > c.ema20 and
        c.close > c.vwap and
        c.rsi >= RSI_LONG and
        c.adx > MIN_ADX and
        c.di_plus > c.di_minus and
        c.close > c.ema9
    )

    confirmed_short = (
        c.ema9 < c.ema20 and
        c.close < c.vwap and
        c.rsi <= RSI_SHORT and
        c.adx > MIN_ADX and
        c.di_minus > c.di_plus and
        c.close < c.ema9
    )

    candidates = []

    if confirmed_long and (cross_long or (
        c.low <= c.ema9 + c.atr * 0.25 and
        c.close > c.open and
        c.close >= c.low + (c.high-c.low) * 0.55
    )):
        candidates.append(("LONG", score(c, "LONG"), "EMA9/EMA20 CROSS" if cross_long else "EMA9 PULLBACK"))

    if confirmed_short and (cross_short or (
        c.high >= c.ema9 - c.atr * 0.25 and
        c.close < c.open and
        c.close <= c.high - (c.high-c.low) * 0.55
    )):
        candidates.append(("SHORT", score(c, "SHORT"), "EMA9/EMA20 CROSS" if cross_short else "EMA9 PULLBACK"))

    # Early momentum: EMA gap shrinking toward crossover while price already
    # has the VWAP + RSI + ADX direction.
    dist = abs(c.ema9 - c.ema20)
    prev_dist = abs(p.ema9 - p.ema20)

    early_long = (
        c.ema9 <= c.ema20 and
        c.ema9 > p.ema9 and
        dist < prev_dist and
        c.close > c.vwap and
        c.rsi >= EARLY_RSI_LONG and
        c.adx > MIN_ADX and
        c.adx >= p.adx and
        c.di_plus >= c.di_minus
    )

    early_short = (
        c.ema9 >= c.ema20 and
        c.ema9 < p.ema9 and
        dist < prev_dist and
        c.close < c.vwap and
        c.rsi <= EARLY_RSI_SHORT and
        c.adx > MIN_ADX and
        c.adx >= p.adx and
        c.di_minus >= c.di_plus
    )

    signal = None
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        side, sc, trigger = candidates[0]
        if sc >= MIN_SCORE:
            signal = {
                "side": side,
                "score": sc,
                "trigger": trigger,
                "spot": float(c.close),
                "ema9": float(c.ema9),
                "ema20": float(c.ema20),
                "vwap": float(c.vwap),
                "rsi": float(c.rsi),
                "adx": float(c.adx),
                "di_plus": float(c.di_plus),
                "di_minus": float(c.di_minus),
                "vol_ratio": float(c.vol_ratio),
                "atr": float(c.atr),
                "open_time": int(c.open_time)
            }

    early = None
    if early_long or early_short:
        side = "LONG" if early_long else "SHORT"
        sc = score(c, side, early=True)
        if sc >= EARLY_MIN_SCORE:
            early = {
                "side": side,
                "score": sc,
                "spot": float(c.close),
                "ema9": float(c.ema9),
                "ema20": float(c.ema20),
                "vwap": float(c.vwap),
                "rsi": float(c.rsi),
                "adx": float(c.adx),
                "di_plus": float(c.di_plus),
                "di_minus": float(c.di_minus),
                "vol_ratio": float(c.vol_ratio),
                "open_time": int(c.open_time)
            }

    return signal, early, "OK"


def analyze_asset(asset):
    signal, early, reason = make_signal(asset)

    if signal:
        refresh_option_market()
        option, option_reason = select_option(
            asset, signal["side"], signal["spot"]
        )

        if option:
            key = (asset, signal["side"], option["symbol"], signal["open_time"])
            cdkey = (asset, signal["side"], "CONF")
            now = time.time()

            if key not in seen and now - last_alert.get(cdkey, 0) >= CONF_COOLDOWN:
                send_confirm(asset, signal, option, option_reason)
                seen.add(key)
                last_alert[cdkey] = now
                stats["confirmed"] += 1
        else:
            stats["blocked_no_option"] += 1
            stats[f"option_reason_{option_reason}"] += 1

    if early:
        cdkey = (asset, early["side"], "EARLY")
        now = time.time()
        if now - last_alert.get(cdkey, 0) >= EARLY_COOLDOWN:
            send_early(asset, early)
            last_alert[cdkey] = now
            stats["early"] += 1


# ============================================================
# DIAGNOSTIC / HEALTH
# ============================================================

@app.get("/")
def home():
    return jsonify({
        "bot": BOT_NAME,
        "status": "running",
        "timeframe": "5m",
        "mandatory": {
            "VWAP": True,
            "EMA_fast": 9,
            "EMA_slow": 21,
            "RSI": 14,
            "ADX": 14
        },
        "spot_assets": sorted(spot_symbols.keys()),
        "option_assets": sorted(option_market.keys()),
        "providers": provider_status,
        "stats": dict(stats)
    })

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "bot": BOT_NAME,
        "timeframe": "5m",
        "mandatory_parameters": "VWAP + EMA9/EMA20 + RSI14 + ADX14",
        "spot_assets": len(spot_symbols),
        "option_assets": len(option_market),
        "bars": {a: len(bars.get(a, [])) for a in spot_symbols},
        "confirmed": stats["confirmed"],
        "early": stats["early"],
        "errors": stats["errors"],
        "providers": provider_status,
        "provider_errors": dict(provider_errors)
    })

@app.get("/test-telegram")
def test_telegram():
    ok = telegram(
        "✅ <b>TELEGRAM TEST OK</b>\n"
        "Crypto 5M VWAP + EMA9/20 + RSI14 + ADX14 bot is connected."
    )
    return jsonify({"telegram_ok": ok})

@app.get("/debug-scan")
def debug_scan():
    snapshot = {}
    for asset in list(spot_symbols.keys()):
        try:
            with lock:
                rows = list(bars.get(asset, []))
            d = build_indicators(rows)
            if d is None:
                snapshot[asset] = {"status": "not_ready", "bars": len(rows)}
                continue
            c = d.iloc[-1]
            snapshot[asset] = {
                "spot": float(c.close),
                "ema9": float(c.ema9),
                "ema20": float(c.ema20),
                "vwap": float(c.vwap),
                "rsi14": float(c.rsi),
                "adx14": float(c.adx),
                "di_plus": float(c.di_plus),
                "di_minus": float(c.di_minus),
                "volume_ratio": float(c.vol_ratio),
                "long_score": score(c, "LONG"),
                "short_score": score(c, "SHORT"),
            }
        except Exception as e:
            snapshot[asset] = {"error": str(e)}

    return jsonify({
        "mandatory": ["VWAP", "EMA9/EMA20", "RSI14", "ADX14"],
        "providers": provider_status,
        "spot_assets": sorted(spot_symbols.keys()),
        "option_assets": sorted(option_market.keys()),
        "stats": dict(stats),
        "snapshot": snapshot,
        "provider_errors": dict(provider_errors)
    })


# ============================================================
# STARTUP
# ============================================================

def startup_message():
    active_spot = ", ".join(sorted(spot_symbols.keys())) or "NONE"
    active_opt = ", ".join(sorted(option_market.keys())) or "NONE"

    telegram(
        f"""🟢 <b>{BOT_NAME} ONLINE</b>

<b>TIMEFRAME:</b> 5 MIN

<b>MANDATORY PARAMETERS</b>
✓ VWAP
✓ EMA9
✓ EMA20
✓ RSI14
✓ ADX14

<b>SPOT DATA</b>
{active_spot}

<b>OPTION DATA</b>
{active_opt}

<b>SIGNALS</b>
⚡ EARLY CALL/PUT WATCH
🟢 CONFIRMED CALL BUY
🔴 CONFIRMED PUT BUY

<b>ENTRY LOGIC</b>
All mandatory indicators must agree for confirmed signals.

No automatic orders are placed."""
    )

def refresh_all():
    global last_option_refresh
    try:
        discover_okx_spot()
    except Exception as e:
        provider_status["okx"] = "ERROR"
        provider_errors["okx_spot"] = str(e)[:300]
        print("OKX spot discovery error:", e)

    try:
        discover_okx_options()
    except Exception as e:
        provider_errors["okx_options"] = str(e)[:300]
        print("OKX options discovery error:", e)

    try:
        discover_deribit_options()
    except Exception as e:
        provider_errors["deribit"] = str(e)[:300]
        print("Deribit discovery error:", e)

    for asset in list(spot_symbols.keys()):
        try:
            n = load_history(asset)
            print("Warmup", asset, n)
        except Exception as e:
            stats["errors"] += 1
            provider_errors[f"warmup_{asset}"] = str(e)[:300]
        time.sleep(0.05)

    last_option_refresh = 0
    try:
        refresh_option_market()
    except Exception as e:
        print("Option refresh error:", e)

def scan_loop():
    last_discovery = 0

    while True:
        try:
            if time.time() - last_discovery >= 30 * 60:
                refresh_all()
                last_discovery = time.time()

            assets = list(spot_symbols.keys())

            for asset in assets:
                try:
                    if refresh_asset(asset):
                        analyze_asset(asset)
                        stats["closed_bars"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    print("Scan error", asset, e)

                time.sleep(SCAN_SLEEP)

        except Exception as e:
            stats["errors"] += 1
            print("Loop error:", e)

        time.sleep(1)

def main():
    print("=" * 72)
    print(BOT_NAME)
    print("=" * 72)
    print("MANDATORY: VWAP + EMA9/EMA20 + RSI14 + ADX14")
    print("TIMEFRAME: 5m")

    refresh_all()

    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=int(os.getenv("PORT", "10000")),
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    ).start()

    startup_message()

    threading.Thread(
        target=scan_loop,
        daemon=True
    ).start()

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
