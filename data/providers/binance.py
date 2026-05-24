import datetime
import time
from enum import Enum

import pandas as pd
import requests


class BinanceAPIAssetType(Enum):
    SPOT = "https://api.binance.com/api/v3/klines"
    FUTURES = "https://fapi.binance.com/fapi/v1/klines"


class BinanceProvider:
    def __init__(self, delay: int = 0.2):
        self.delay = delay

    def get_pairs(self) -> dict:
        """
        Get all Binance USDT trading pairs
        """
        resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
        info = resp.json()
        pairs = {}
        for s in info["symbols"]:
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
                base = s["baseAsset"].upper()
                pairs[base] = s["symbol"]
        return pairs

    def get_futures(self) -> dict:
        """
        Get all Binance USDT-M perpetual futures pairs
        """
        resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=15)
        info = resp.json()
        pairs = {}
        for s in info["symbols"]:
            if s["status"] == "TRADING" and s["contractType"] == "PERPETUAL":
                base = s["baseAsset"].upper()
                pairs[base] = s["symbol"]
        return pairs

    def get_futures_w_spot(self) -> pd.DataFrame:
        """
        Return coins that have both a spot USDT pair and a USDT-M perpetual future.
        Columns: base_asset, spot_symbol, futures_symbol
        """
        spot = self.get_pairs()
        futures = self.get_futures()
        overlap = sorted(set(spot) & set(futures))
        return pd.DataFrame(
            [
                {"base_asset": c, "spot_symbol": spot[c], "futures_symbol": futures[c]}
                for c in overlap
            ]
        )

    def get_klines(self, symbol, source: str = "SPOT", days: int = 730):
        """
        Fetch daily OHLCV from Binance, paginating if needed
        """
        all_klines = []
        end_time = int(datetime.datetime.now().timestamp() * 1000)

        while len(all_klines) < days:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "limit": 1000,
                "endTime": end_time,
            }
            url = BinanceAPIAssetType[source].value
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"    Binance error for {symbol}: {resp.status_code}")
                break
            klines = resp.json()
            if not klines:
                break
            all_klines = klines + all_klines
            end_time = klines[0][0] - 1
            time.sleep(self.delay)

        # Parse into dicts
        result = []
        for k in all_klines[-days:]:
            result.append(
                {
                    "symbol": symbol,
                    "date": pd.to_datetime(k[0], unit="ms").strftime("%Y-%m-%d"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume_base": float(k[5]),
                    "volume_usdt": float(k[7]),
                    "trades": int(k[8]),
                }
            )
        return result
