"""
RVOL Momentum Strategy — Live Trader (Cross Margin)
Runs once daily (after UTC midnight candle close).
Uses Binance Cross Margin for both long and short legs.
Wider universe than futures — any cross-margin listed coin can be shorted.

Usage:
    python live_trader_margin.py            # dry-run (prints orders, executes nothing)
    python live_trader_margin.py --live     # live mode (executes real orders)

See live_trader.py for the USDT-M Futures equivalent.
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

CAPITAL = 1_000.0  # total USDT allocated to this strategy
K = 5  # holding period in days (JT tranches)
SHORT_SMA = 5  # RVOL numerator window
LONG_SMA = 30  # RVOL denominator window
N_LONGS = 5  # top N coins to go long
N_SHORTS = 5  # bottom N coins to go short
MIN_NOTIONAL = 10.0  # skip trades smaller than this (USDT)
FEE_RATE = 0.001  # Binance margin taker fee (0.1%)

STABLECOINS = {
    "USDT",
    "USDC",
    "BUSD",
    "DAI",
    "TUSD",
    "USDP",
    "USDD",
    "GUSD",
    "FRAX",
    "LUSD",
    "SUSD",
    "CRVUSD",
    "PYUSD",
    "FDUSD",
    "USDE",
    "EURC",
    "EURT",
    "GHO",
    "ALUSD",
    "MIM",
    "UST",
    "USTC",
    "USDB",
    "USD0",
    "USDS",
    "USDJ",
    "CUSD",
    "RSV",
    "PAX",
}
WRAPPED_TOKENS = {
    "WBTC",
    "WETH",
    "STETH",
    "CBETH",
    "RETH",
    "WBNB",
    "WMATIC",
    "WSOL",
    "WTRX",
    "HBTC",
    "RENBTC",
    "TBTC",
    "SBTC",
    "BETH",
    "METH",
    "MSOL",
    "BNSOL",
    "JITOSOL",
}

API_BASE = "https://api.binance.com"
CG_BASE = "https://api.coingecko.com/api/v3"
STATE_FILE     = Path(__file__).parent / "state_margin.json"
TRADES_FILE    = Path(__file__).parent / "trades_margin.json"
SNAPSHOTS_FILE = Path(__file__).parent / "snapshots_margin.json"


# ── Binance Cross Margin API Client ───────────────────────────────────────────


class MarginClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    def get(self, path: str, params: dict = None, signed: bool = False) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        resp = self.session.get(f"{API_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        resp = self.session.post(f"{API_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Public endpoints ──────────────────────────────────────────────────────

    def exchange_info(self) -> dict:
        return self.get("/api/v3/exchangeInfo")

    def margin_pairs(self) -> list:
        return self.get("/sapi/v1/margin/allPairs")

    def klines(self, symbol: str, interval: str = "1d", limit: int = 35) -> list:
        return self.get(
            "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit}
        )

    def ticker_price(self, symbol: str) -> dict:
        return self.get("/api/v3/ticker/price", {"symbol": symbol})

    # ── Private endpoints ─────────────────────────────────────────────────────

    def margin_account(self) -> dict:
        return self.get("/sapi/v1/margin/account", signed=True)

    def place_market_order(self, symbol: str, side: str, qty: float) -> dict:
        # BUY  → AUTO_REPAY: repays any borrowed (short) position automatically
        # SELL → MARGIN_BUY: auto-borrows the asset if needed to sell short
        side_effect = "AUTO_REPAY" if side == "BUY" else "MARGIN_BUY"
        return self.post(
            "/sapi/v1/margin/order",
            {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
                "sideEffectType": side_effect,
            },
        )


# ── Strategy State ────────────────────────────────────────────────────────────


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"tranches": {}}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_trade(record: dict) -> None:
    data = {"trades": []}
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            data = json.load(f)
    data["trades"].append(record)
    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def append_snapshot(date: str, portfolio_value: float) -> None:
    data = {"snapshots": []}
    if SNAPSHOTS_FILE.exists():
        with open(SNAPSHOTS_FILE) as f:
            data = json.load(f)
    # overwrite today's entry if it already exists
    data["snapshots"] = [s for s in data["snapshots"] if s["date"] != date]
    data["snapshots"].append({"date": date, "portfolio_value": round(portfolio_value, 4)})
    data["snapshots"].sort(key=lambda s: s["date"])
    with open(SNAPSHOTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Universe Construction ─────────────────────────────────────────────────────


def get_margin_universe(client: MarginClient, quote: str = "USDC") -> dict:
    """
    Returns dict of { base_symbol -> "BASEUSDT" } for coins that are:
    - In CoinGecko top 250 by market cap
    - Not a stablecoin or wrapped token
    - Listed as a cross-margin tradeable USDT pair on Binance
    """
    # All cross-margin pairs from Binance
    pairs = client.margin_pairs()
    margin_map = {
        p["base"].upper(): p["symbol"]
        for p in pairs
        if p["quote"].upper() == quote and p.get("isMarginTrade", False)
    }

    # CoinGecko top 250 by market cap
    resp = requests.get(
        f"{CG_BASE}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
        },
        timeout=15,
    )
    resp.raise_for_status()
    cg_coins = resp.json()

    excluded = STABLECOINS | WRAPPED_TOKENS
    universe = {}
    for coin in cg_coins:
        sym = coin["symbol"].upper()
        if sym in excluded:
            continue
        if sym in margin_map:
            universe[sym] = margin_map[sym]
        if len(universe) >= 100:
            break

    print(f"  Universe: {len(universe)} coins with cross-margin USDT pairs")
    return universe


# ── Signal Computation ────────────────────────────────────────────────────────


def compute_signal(client: MarginClient, universe: dict) -> dict | None:
    """
    Computes RVOL = SHORT_SMA(volume) / LONG_SMA(volume) for each coin.
    Returns {'top': [symbols], 'bot': [symbols]} or None if insufficient data.
    """
    rvol = {}
    for sym, margin_sym in universe.items():
        try:
            klines = client.klines(margin_sym, limit=LONG_SMA + 5)
            # k[7] is quote asset volume (USDT); exclude the still-forming candle
            volumes = [float(k[7]) for k in klines[:-1]]
            if len(volumes) < LONG_SMA:
                continue
            short_mean = sum(volumes[-SHORT_SMA:]) / SHORT_SMA
            long_mean = sum(volumes[-LONG_SMA:]) / LONG_SMA
            if long_mean > 0:
                rvol[sym] = short_mean / long_mean
            time.sleep(0.05)
        except Exception as e:
            print(f"    Warning: could not fetch {margin_sym}: {e}")

    if len(rvol) < N_LONGS + N_SHORTS:
        print(
            f"  Only {len(rvol)} coins with valid RVOL — need at least {N_LONGS + N_SHORTS}"
        )
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


def get_current_weights(client: MarginClient) -> dict:
    """
    Reads cross-margin account and converts each asset's net holding to a weight.
    Positive netAsset = long position, negative netAsset = short position.
    """
    account = client.margin_account()
    weights = {}
    for asset in account["userAssets"]:
        sym = asset["asset"].upper()
        if sym == "USDT":
            continue
        net = float(asset["netAsset"])
        if abs(net) < 1e-8:
            continue
        try:
            price = float(client.ticker_price(f"{sym}USDT")["price"])
        except Exception:
            continue
        weights[sym] = (net * price) / CAPITAL
    return weights


# ── Lot Size Helpers ──────────────────────────────────────────────────────────

_lot_cache: dict[str, float] = {}


def get_step_size(client: MarginClient, symbol: str) -> float:
    if symbol in _lot_cache:
        return _lot_cache[symbol]
    info = client.exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    _lot_cache[symbol] = step
                    return step
    return 0.001  # fallback


def round_qty(qty: float, step: float) -> float:
    precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(round(qty / step) * step, precision)


# ── Rebalancing ───────────────────────────────────────────────────────────────


def rebalance(
    client: MarginClient,
    universe: dict,
    target: dict,
    current: dict,
    dry_run: bool,
) -> None:
    """Compute weight deltas and place margin market orders to close the gap."""
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

        margin_sym = universe[sym]
        price = float(client.ticker_price(margin_sym)["price"])
        qty = notional / price
        step = get_step_size(client, margin_sym)
        qty = round_qty(qty, step)

        if qty <= 0:
            continue

        side = "BUY" if delta > 0 else "SELL"
        side_effect = "AUTO_REPAY" if side == "BUY" else "MARGIN_BUY"
        orders.append((sym, margin_sym, side, side_effect, qty, notional))

    if not orders:
        print("  No rebalancing needed.")
        return

    print(f"\n  {'Symbol':<10} {'Side':<6} {'Effect':<12} {'Qty':>12} {'Notional':>12}")
    print("  " + "-" * 56)
    for sym, margin_sym, side, side_effect, qty, notional in orders:
        print(f"  {sym:<10} {side:<6} {side_effect:<12} {qty:>12g} ${notional:>10.2f}")
        if not dry_run:
            try:
                result = client.place_market_order(margin_sym, side, qty)
                print(f"    → Order ID {result['orderId']} filled")
                append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "margin_symbol": margin_sym,
                    "side": side,
                    "side_effect": side_effect,
                    "qty": qty,
                    "notional": round(notional, 4),
                    "price": price,
                    "order_id": result["orderId"],
                })
            except Exception as e:
                print(f"    → ERROR placing order for {margin_sym}: {e}")
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
    api_key = os.environ["BNB_KEY"]
    api_secret = os.environ["BNB_SECRET"]

    mode = "DRY RUN" if dry_run else "LIVE"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n{'='*55}")
    print(f"  RVOL Trader (Margin) — {today} UTC  [{mode}]")
    print(f"  Capital: ${CAPITAL:,.0f}  |  K={K}  |  L={N_LONGS}/S={N_SHORTS}")
    print(f"{'='*55}")

    client = MarginClient(api_key, api_secret)
    state = load_state()

    # 1. Universe
    print("\n[1] Building universe...")
    universe = get_margin_universe(client)

    # 2. Signal
    print("\n[2] Computing RVOL signal...")
    signal = compute_signal(client, universe)
    if signal is None:
        print("  No valid signal today — skipping run.")
        return

    # 3. Add new tranche (skip if already ran today)
    if today not in state["tranches"]:
        state["tranches"][today] = {
            "top": signal["top"],
            "bot": signal["bot"],
            "days_left": K,
        }
        print(f"  Added tranche for {today}  ({len(state['tranches'])} active)")
    else:
        print(f"  Tranche for {today} already exists — skipping add.")

    # 4. Target weights
    target = compute_target_weights(state)
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

    # 7. Portfolio snapshot
    try:
        account = client.margin_account()
        usdt_free = sum(
            float(a["free"]) + float(a["locked"])
            for a in account["userAssets"]
            if a["asset"].upper() == "USDT"
        )
        position_value = 0.0
        for a in account["userAssets"]:
            sym = a["asset"].upper()
            if sym == "USDT":
                continue
            net = float(a["netAsset"])
            if abs(net) < 1e-8:
                continue
            try:
                price = float(client.ticker_price(f"{sym}USDT")["price"])
                position_value += net * price
            except Exception:
                pass
        portfolio_value = usdt_free + position_value
        append_snapshot(today, portfolio_value)
        print(f"\n[6] Portfolio snapshot: ${portfolio_value:,.2f}")
    except Exception as e:
        print(f"\n[6] Snapshot failed: {e}")

    # 8. Expire tranches and save
    print("\n[7] Ticking tranches...")
    tick_tranches(state)
    save_state(state)
    print(f"  State saved → {STATE_FILE}")
    print(f"  Active tranches remaining: {len(state['tranches'])}")


if __name__ == "__main__":
    live = "--live" in sys.argv
    if live:
        confirm = input(
            "\n  ⚠️  LIVE mode — real orders will be placed. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("  Aborted.")
            sys.exit(0)
    run(dry_run=not live)
