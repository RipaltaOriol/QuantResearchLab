"""
RVOL Momentum Strategy — Live Trader
Runs once daily (after UTC midnight candle close).
Uses Binance USDT-M Perpetual Futures for both long and short legs.

Usage:
    python live_trader.py            # dry-run (prints orders, executes nothing)
    python live_trader.py --live     # live mode (executes real orders)
"""

import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────

CAPITAL        = 1_000.0   # total USDT allocated to this strategy
K              = 5         # holding period in days (JT tranches)
SHORT_SMA      = 5         # RVOL numerator window
LONG_SMA       = 30        # RVOL denominator window
N_LONGS        = 5         # top N coins to go long
N_SHORTS       = 5         # bottom N coins to go short
MIN_NOTIONAL   = 10.0      # skip trades smaller than this (USDT)
FEE_RATE       = 0.0004    # Binance futures taker fee (0.04%)

STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USDD", "GUSD",
    "FRAX", "LUSD", "SUSD", "CRVUSD", "PYUSD", "FDUSD", "USDE",
    "EURC", "EURT", "GHO", "ALUSD", "MIM", "UST", "USTC",
    "USDB", "USD0", "USDS", "USDJ", "CUSD", "RSV", "PAX",
}
WRAPPED_TOKENS = {
    "WBTC", "WETH", "STETH", "CBETH", "RETH", "WBNB", "WMATIC",
    "WSOL", "WTRX", "HBTC", "RENBTC", "TBTC", "SBTC", "BETH",
    "METH", "MSOL", "BNSOL", "JITOSOL",
}

FAPI_BASE  = "https://fapi.binance.com"
CG_BASE    = "https://api.coingecko.com/api/v3"
STATE_FILE = Path(__file__).parent / "state.json"


# ── Binance Futures API Client ────────────────────────────────────────────────

class FuturesClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.session    = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig   = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    def get(self, path: str, params: dict = None, signed: bool = False) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        resp = self.session.get(f"{FAPI_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        resp = self.session.post(f"{FAPI_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Public endpoints ──────────────────────────────────────────────────────

    def exchange_info(self) -> dict:
        return self.get("/fapi/v1/exchangeInfo")

    def klines(self, symbol: str, interval: str = "1d", limit: int = 35) -> list:
        return self.get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })

    def ticker_price(self, symbol: str) -> dict:
        return self.get("/fapi/v1/ticker/price", {"symbol": symbol})

    # ── Private endpoints ─────────────────────────────────────────────────────

    def positions(self) -> list:
        return self.get("/fapi/v2/positionRisk", signed=True)

    def balance(self) -> list:
        return self.get("/fapi/v2/balance", signed=True)

    def place_market_order(self, symbol: str, side: str, qty: float) -> dict:
        return self.post("/fapi/v1/order", {
            "symbol":   symbol,
            "side":     side,        # "BUY" or "SELL"
            "type":     "MARKET",
            "quantity": qty,
        })


# ── Strategy State ────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"tranches": {}}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Universe Construction ─────────────────────────────────────────────────────

def get_futures_universe(client: FuturesClient) -> dict:
    """
    Returns dict of { base_symbol -> futures_symbol } for coins that are:
    - In CoinGecko top 250 by market cap
    - Not a stablecoin or wrapped token
    - Have an active USDT-M perpetual futures contract on Binance
    """
    # Fetch active USDT-M perpetual futures symbols from Binance
    info = client.exchange_info()
    futures_pairs = {
        s["baseAsset"].upper(): s["symbol"]
        for s in info["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    }

    # Fetch CoinGecko top 250
    resp = requests.get(f"{CG_BASE}/coins/markets", params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
    }, timeout=15)
    resp.raise_for_status()
    cg_coins = resp.json()

    excluded = STABLECOINS | WRAPPED_TOKENS
    universe = {}
    for coin in cg_coins:
        sym = coin["symbol"].upper()
        if sym in excluded:
            continue
        if sym in futures_pairs:
            universe[sym] = futures_pairs[sym]
        if len(universe) >= 100:
            break

    print(f"  Universe: {len(universe)} coins with USDT-M futures")
    return universe


# ── Signal Computation ────────────────────────────────────────────────────────

def compute_signal(client: FuturesClient, universe: dict) -> dict | None:
    """
    Computes RVOL = SHORT_SMA(volume) / LONG_SMA(volume) for each coin.
    Returns {'top': [symbols], 'bot': [symbols]} or None if insufficient data.
    """
    rvol = {}
    for sym, futures_sym in universe.items():
        try:
            klines = client.klines(futures_sym, limit=LONG_SMA + 5)
            # klines[i][7] is quote asset volume (USDT)
            volumes = [float(k[7]) for k in klines[:-1]]  # exclude incomplete candle
            if len(volumes) < LONG_SMA:
                continue
            short_mean = sum(volumes[-SHORT_SMA:]) / SHORT_SMA
            long_mean  = sum(volumes[-LONG_SMA:])  / LONG_SMA
            if long_mean > 0:
                rvol[sym] = short_mean / long_mean
            time.sleep(0.05)
        except Exception as e:
            print(f"    Warning: could not fetch {futures_sym}: {e}")

    if len(rvol) < N_LONGS + N_SHORTS:
        print(f"  Only {len(rvol)} coins with valid RVOL — need at least {N_LONGS + N_SHORTS}")
        return None

    sorted_rvol = sorted(rvol.items(), key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in sorted_rvol[:N_LONGS]]
    bot = [sym for sym, _ in sorted_rvol[-N_SHORTS:]]

    print(f"  Signal  | Long:  {top}")
    print(f"          | Short: {bot}")
    return {"top": top, "bot": bot}


# ── Weight Computation ────────────────────────────────────────────────────────

def compute_target_weights(state: dict) -> dict:
    """
    Sums active tranches (each scaled by 1/K) to get net target weight per coin.
    Positive weight = long, negative = short.
    """
    weights: dict[str, float] = {}
    for tranche in state["tranches"].values():
        for sym in tranche["top"]:
            weights[sym] = weights.get(sym, 0.0) + (1.0 / N_LONGS) / K
        for sym in tranche["bot"]:
            weights[sym] = weights.get(sym, 0.0) - (1.0 / N_SHORTS) / K
    return weights


# ── Current Positions ─────────────────────────────────────────────────────────

def get_current_weights(client: FuturesClient) -> dict:
    """
    Reads open futures positions and converts to weight = notional / CAPITAL.
    Positive = long, negative = short.
    """
    positions = client.positions()
    weights = {}
    for p in positions:
        amt = float(p["positionAmt"])
        if amt == 0:
            continue
        price = float(p["entryPrice"])
        if price == 0:
            continue
        sym = p["symbol"].replace("USDT", "")
        notional = amt * price          # positive if long, negative if short
        weights[sym] = notional / CAPITAL
    return weights


# ── Lot Size Helpers ──────────────────────────────────────────────────────────

_lot_cache: dict[str, float] = {}

def get_step_size(client: FuturesClient, futures_sym: str) -> float:
    if futures_sym in _lot_cache:
        return _lot_cache[futures_sym]
    info = client.exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == futures_sym:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    _lot_cache[futures_sym] = step
                    return step
    return 0.001  # fallback


def round_qty(qty: float, step: float) -> float:
    precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(round(qty / step) * step, precision)


# ── Rebalancing ───────────────────────────────────────────────────────────────

def rebalance(
    client: FuturesClient,
    universe: dict,
    target: dict,
    current: dict,
    dry_run: bool,
) -> None:
    """Compute weight deltas and place market orders to close the gap."""
    all_coins = set(target.keys()) | set(current.keys())
    orders = []

    for sym in all_coins:
        if sym not in universe:
            continue
        t = target.get(sym, 0.0)
        c = current.get(sym, 0.0)
        delta = t - c

        notional = abs(delta) * CAPITAL
        if notional < MIN_NOTIONAL:
            continue

        futures_sym = universe[sym]
        price = float(client.ticker_price(futures_sym)["price"])
        qty   = notional / price
        step  = get_step_size(client, futures_sym)
        qty   = round_qty(qty, step)

        if qty <= 0:
            continue

        side = "BUY" if delta > 0 else "SELL"
        orders.append((sym, futures_sym, side, qty, notional))

    if not orders:
        print("  No rebalancing needed.")
        return

    print(f"\n  {'Symbol':<10} {'Side':<6} {'Qty':>12} {'Notional':>12}")
    print("  " + "-" * 44)
    for sym, futures_sym, side, qty, notional in orders:
        print(f"  {sym:<10} {side:<6} {qty:>12g} ${notional:>10.2f}")
        if not dry_run:
            try:
                result = client.place_market_order(futures_sym, side, qty)
                print(f"    → Order ID {result['orderId']} filled")
            except Exception as e:
                print(f"    → ERROR placing order for {futures_sym}: {e}")
            time.sleep(0.1)


# ── Tranche Lifecycle ─────────────────────────────────────────────────────────

def tick_tranches(state: dict) -> None:
    """Decrement days_left and remove expired tranches."""
    expired = [d for d, t in state["tranches"].items() if t["days_left"] <= 1]
    for d in expired:
        print(f"  Expired tranche from {d}")
        del state["tranches"][d]
    for t in state["tranches"].values():
        t["days_left"] -= 1


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run(dry_run: bool = True) -> None:
    load_dotenv()
    api_key    = os.environ["BNB_KEY"]
    api_secret = os.environ["BNB_SECRET"]

    mode = "DRY RUN" if dry_run else "LIVE"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n{'='*55}")
    print(f"  RVOL Trader — {today} UTC  [{mode}]")
    print(f"  Capital: ${CAPITAL:,.0f}  |  K={K}  |  L={N_LONGS}/S={N_SHORTS}")
    print(f"{'='*55}")

    client = FuturesClient(api_key, api_secret)
    state  = load_state()

    # 1. Universe
    print("\n[1] Building universe...")
    universe = get_futures_universe(client)

    # 2. Signal
    print("\n[2] Computing RVOL signal...")
    signal = compute_signal(client, universe)
    if signal is None:
        print("  No valid signal today — skipping run.")
        return

    # 3. Add new tranche (skip if already ran today)
    if today not in state["tranches"]:
        state["tranches"][today] = {
            "top":       signal["top"],
            "bot":       signal["bot"],
            "days_left": K,
        }
        print(f"  Added tranche for {today}  ({len(state['tranches'])} active)")
    else:
        print(f"  Tranche for {today} already exists — skipping add.")

    # 4. Target weights
    target  = compute_target_weights(state)
    print(f"\n[3] Target weights ({len(target)} coins):")
    for sym, w in sorted(target.items(), key=lambda x: -abs(x[1])):
        bar = "+" if w > 0 else "-"
        print(f"      {sym:<8} {w:+.4f}  ({bar})")

    # 5. Current positions
    print("\n[4] Current positions...")
    current = get_current_weights(client)
    if current:
        for sym, w in current.items():
            print(f"      {sym:<8} {w:+.4f}")
    else:
        print("      (none)")

    # 6. Rebalance
    print(f"\n[5] Rebalancing {'(DRY RUN)' if dry_run else '(LIVE)'}...")
    rebalance(client, universe, target, current, dry_run)

    # 7. Expire tranches and save
    print("\n[6] Ticking tranches...")
    tick_tranches(state)
    save_state(state)
    print(f"  State saved → {STATE_FILE}")
    print(f"  Active tranches remaining: {len(state['tranches'])}")


if __name__ == "__main__":
    live = "--live" in sys.argv
    if live:
        confirm = input("\n  ⚠️  LIVE mode — real orders will be placed. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("  Aborted.")
            sys.exit(0)
    run(dry_run=not live)
