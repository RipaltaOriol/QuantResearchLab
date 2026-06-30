import os
import time
from typing import List, Set
import pandas as pd
import requests
from dotenv import load_dotenv

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

API_KEY = os.environ["COINGECKO_KEY"]
PRO_URL = "https://pro-api.coingecko.com/api/v3"
class CoingeckoProvider:
    def __init__(self, delay: int = 7):
        self.delay = delay

    def get_coins(self) -> dict:
        headers = {"x-cg-pro-api-key": API_KEY}
        res = requests.get(f"{PRO_URL}/coins/list", headers=headers)
        res.raise_for_status()
        return {c["id"]: c["symbol"] for c in res.json()}

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
    
    def get_coins_by_category(self, category_id: str) -> Set:
        """
        Gets a list of coins from the given category
        """
        symbols = set()
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "category": category_id,
                    "per_page": 250,
                    "page": 1,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                for c in resp.json():
                    symbols.add(c["symbol"].upper())
                print(
                    f"  Loaded {len(symbols)} {category_id} symbols"
                )
            else:
                print(
                    f"  CoinGecko {category_id} fetch failed ({resp.status_code}), using hardcoded list"
                )
        except Exception as e:
            print(f"  CoinGecko {category_id} fetch error: {e}")
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

    def get_coin_marketdata(self, coin_id, start: str, end: str, currency: str = "usd", symbol: str = None, is_price: bool = False) -> pd.DataFrame:
        """
        Improvement over the previus API 'get_coin_data'. Reconcile results
        """
        headers = {"x-cg-pro-api-key": API_KEY}
        params = {
            "vs_currency": currency,
            "from": start,
            "to": end,
        }
        res = requests.get(f"{PRO_URL}/coins/{coin_id}/market_chart/range",
                           headers=headers,
                           params=params
                            )

        if res.status_code == 429:
            time.sleep(10)
            return self.get_coin_marketdata(coin_id, start, end, currency)
        
        if res.status_code != 200:
            print(f"Skipping {coin_id}: {res.status_code}")
            return pd.DataFrame()
    
        data = res.json()

        prcs = pd.DataFrame(data.get("prices", []), columns=["timestamp", "prices"])
        caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp", "market_cap"])
        vols = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])

        if caps.empty or vols.empty:
            return caps
        
        df = caps.merge(vols, on="timestamp", how="outer")

        if is_price:
            df = df.merge(prcs, on='timestamp', how="outer")

        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
        df["coin_id"] = coin_id

        if symbol:
            df['symbol'] = symbol

        if is_price:
            return df[["date", "coin_id", "market_cap", "prices"]]
        return df[["date", "coin_id", "market_cap", "volume"]]