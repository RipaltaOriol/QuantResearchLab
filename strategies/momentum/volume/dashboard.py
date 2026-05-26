"""
RVOL Strategy Dashboard
Reads from state_margin.json, trades_margin.json, snapshots_margin.json
and the live Binance Cross Margin account.

Run:
    streamlit run dashboard.py
    # Docker / remote:
    streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from live_trader_margin import (
    CAPITAL,
    K,
    N_LONGS,
    N_SHORTS,
    SNAPSHOTS_FILE,
    STATE_FILE,
    TRADES_FILE,
    MarginClient,
    compute_target_weights,
    get_current_weights,
    load_state,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RVOL Strategy",
    page_icon="📈",
    layout="wide",
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="auto_refresh")

# ── Load credentials ──────────────────────────────────────────────────────────

load_dotenv()
_api_key    = os.environ.get("BNB_KEY", "")
_api_secret = os.environ.get("BNB_SECRET", "")

@st.cache_resource
def get_client() -> MarginClient:
    return MarginClient(_api_key, _api_secret)


client = get_client()

# ── Helper loaders ────────────────────────────────────────────────────────────


def load_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    with open(TRADES_FILE) as f:
        return json.load(f).get("trades", [])


def load_snapshots() -> list[dict]:
    if not SNAPSHOTS_FILE.exists():
        return []
    with open(SNAPSHOTS_FILE) as f:
        return json.load(f).get("snapshots", [])


# ── Header metrics ────────────────────────────────────────────────────────────

st.title("RVOL Momentum Strategy — Dashboard")
st.caption(f"Capital: ${CAPITAL:,.0f} · K={K} · {N_LONGS}L / {N_SHORTS}S · auto-refresh every 60 s")

snapshots = load_snapshots()
state     = load_state()

latest_value   = snapshots[-1]["portfolio_value"] if snapshots else CAPITAL
last_snap_date = snapshots[-1]["date"]            if snapshots else "—"
pnl_usdt       = latest_value - CAPITAL
pnl_pct        = (pnl_usdt / CAPITAL) * 100
n_active       = len(state["tranches"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capital",          f"${CAPITAL:,.0f}")
c2.metric("Total P&L",        f"${pnl_usdt:+,.2f}", f"{pnl_pct:+.2f}%")
c3.metric("Active Tranches",  f"{n_active} / {K}")
c4.metric("Last Snapshot",    last_snap_date)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_pos, tab_weights, tab_tranches, tab_trades, tab_returns = st.tabs(
    ["Positions", "Target Weights", "Tranches", "Trade History", "Returns"]
)

# ── Tab 1: Positions ──────────────────────────────────────────────────────────

with tab_pos:
    st.subheader("Live Positions (Cross Margin)")
    with st.spinner("Fetching account…"):
        try:
            account = client.margin_account()
            rows = []
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
                    price = 0.0
                value  = net * price
                weight = value / CAPITAL
                rows.append({
                    "Symbol":    sym,
                    "Net Asset": round(net, 6),
                    "Price":     price,
                    "Value (USDT)": round(value, 2),
                    "Weight":    round(weight, 4),
                    "Direction": "Long" if net > 0 else "Short",
                })

            if rows:
                df = pd.DataFrame(rows).sort_values("Value (USDT)", key=abs, ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Horizontal bar chart
                colors = ["#26a69a" if d == "Long" else "#ef5350" for d in df["Direction"]]
                fig = go.Figure(go.Bar(
                    x=df["Weight"],
                    y=df["Symbol"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{w:+.2%}" for w in df["Weight"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Current Weights",
                    xaxis_title="Weight",
                    height=max(300, len(df) * 35),
                    margin=dict(l=0, r=40, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No open positions.")
        except Exception as e:
            st.error(f"Could not fetch account: {e}")

# ── Tab 2: Target Weights ─────────────────────────────────────────────────────

with tab_weights:
    st.subheader("Target vs Current Weights")
    target  = compute_target_weights(state)

    with st.spinner("Fetching current weights…"):
        try:
            current = get_current_weights(client)
        except Exception:
            current = {}

    all_syms = sorted(set(target) | set(current), key=lambda s: abs(target.get(s, 0)), reverse=True)
    if all_syms:
        rows = []
        for sym in all_syms:
            t = target.get(sym, 0.0)
            c = current.get(sym, 0.0)
            rows.append({"Symbol": sym, "Target": round(t, 4), "Current": round(c, 4), "Delta": round(t - c, 4)})
        df = pd.DataFrame(rows)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Target",  x=df["Symbol"], y=df["Target"],  marker_color="#42a5f5"))
        fig.add_trace(go.Bar(name="Current", x=df["Symbol"], y=df["Current"], marker_color="#ab47bc"))
        fig.update_layout(
            barmode="group",
            title="Target vs Current Weight per Coin",
            yaxis_title="Weight",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No target or current positions.")

# ── Tab 3: Tranches ───────────────────────────────────────────────────────────

with tab_tranches:
    st.subheader("Active Tranches")
    tranches = state.get("tranches", {})
    if tranches:
        rows = []
        for date, t in sorted(tranches.items(), reverse=True):
            rows.append({
                "Date":      date,
                "Longs":     ", ".join(t["top"]),
                "Shorts":    ", ".join(t["bot"]),
                "Days Left": t["days_left"],
                "Progress":  t["days_left"] / K,
            })
        df = pd.DataFrame(rows)

        for _, row in df.iterrows():
            with st.container():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"**{row['Date']}** &nbsp; "
                        f"🟢 {row['Longs']} &nbsp;|&nbsp; "
                        f"🔴 {row['Shorts']}"
                    )
                with col_b:
                    st.progress(row["Progress"], text=f"{row['Days Left']}d left")
    else:
        st.info("No active tranches.")

# ── Tab 4: Trade History ──────────────────────────────────────────────────────

with tab_trades:
    st.subheader("Trade History")
    trades = load_trades()
    if trades:
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")

        col_sym, col_side = st.columns(2)
        symbols = ["All"] + sorted(df["symbol"].unique().tolist())
        filter_sym  = col_sym.selectbox("Symbol",  symbols)
        filter_side = col_side.selectbox("Side",   ["All", "BUY", "SELL"])

        if filter_sym  != "All": df = df[df["symbol"] == filter_sym]
        if filter_side != "All": df = df[df["side"]   == filter_side]

        display_cols = ["timestamp", "symbol", "side", "side_effect", "qty", "price", "notional", "order_id"]
        st.dataframe(df[display_cols].sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet. Trades are logged only in live (--live) mode.")

# ── Tab 5: Returns ────────────────────────────────────────────────────────────

with tab_returns:
    st.subheader("Portfolio Equity Curve")
    if snapshots:
        df = pd.DataFrame(snapshots)
        df["date"]         = pd.to_datetime(df["date"])
        df["daily_return"] = df["portfolio_value"].pct_change() * 100
        df["cum_return"]   = (df["portfolio_value"] / CAPITAL - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["portfolio_value"],
            mode="lines+markers",
            line=dict(color="#42a5f5", width=2),
            name="Portfolio Value",
        ))
        fig.add_hline(y=CAPITAL, line_dash="dash", line_color="gray", annotation_text="Capital")
        fig.update_layout(
            yaxis_title="USDT",
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Returns")
        display = df[["date", "portfolio_value", "daily_return", "cum_return"]].copy()
        display.columns = ["Date", "Portfolio Value", "Daily Return (%)", "Cumulative Return (%)"]
        display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No snapshots yet. Snapshots are recorded each time the strategy script runs.")
