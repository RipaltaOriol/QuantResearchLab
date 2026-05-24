import time
from typing import List, Set

import pandas as pd
import requests

STABLECOINS_HARDCODED = {
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


class CoingeckoProvider:
    def __init__(self, delay: int = 7):
        self.delay = delay

    def get_top_coins(self, n: int = 250) -> List:
        """
        Fetch top coins by market cap from CoinGecko
        """
        coins = []
        pages_needed = (n // 250) + 1
        for page in range(1, pages_needed + 1):
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                coins.extend(resp.json())
            else:
                print(f"  Warning: page {page} returned {resp.status_code}")
            time.sleep(self.delay)
        return coins

    def get_stablecoins(self) -> Set:
        """
        Build stablecoin set from hardcoded + CoinGecko category
        """

        symbols = set(STABLECOINS_HARDCODED)
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "category": "stablecoins",
                    "per_page": 250,
                    "page": 1,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                for c in resp.json():
                    symbols.add(c["symbol"].upper())
                print(
                    f"  Loaded {len(symbols)} stablecoin symbols (hardcoded + CoinGecko)"
                )
            else:
                print(
                    f"  CoinGecko stablecoin fetch failed ({resp.status_code}), using hardcoded list"
                )
        except Exception as e:
            print(f"  CoinGecko stablecoin fetch error: {e}, using hardcoded list")
        return symbols

    def get_coin_data(self, coin_id, days: int = 365):
        """
        Fetch daily price, market cap, volume from CoinGecko
        """
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=15,
        )
        if resp.status_code == 429:
            print(f"    Rate limited on {coin_id}, waiting 60s...")
            time.sleep(60)
            return self.get_coin_data(coin_id, days)
        if resp.status_code != 200:
            print(f"    CoinGecko error for {coin_id}: {resp.status_code}")
            return None
        data = resp.json()

        result = []
        for i in range(len(data.get("market_caps", []))):
            ts = data["market_caps"][i][0]
            date_str = pd.to_datetime(ts, unit="ms").strftime("%Y-%m-%d")
            result.append(
                {
                    "date": date_str,
                    "cg_price": (
                        data["prices"][i][1] if i < len(data["prices"]) else None
                    ),
                    "cg_market_cap": data["market_caps"][i][1],
                    "cg_volume": (
                        data["total_volumes"][i][1]
                        if i < len(data["total_volumes"])
                        else None
                    ),
                }
            )
        return result
