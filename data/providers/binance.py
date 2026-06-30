import datetime
import time
from enum import Enum
from typing import List
import pandas as pd
import requests


class BinanceAPIAssetType(Enum):
    SPOT = "https://api.binance.com/api/v3/klines"
    FUTURES = "https://fapi.binance.com/fapi/v1/klines"


class BinanceProvider:
    def __init__(self, delay: int = 0.2):
        self.delay = delay

    def get_pairs(self, is_trading = False, permissions: str = None ) -> List:
        """
        Get all Binance USDT trading pairs
        """
        params = {}
        if permissions:
            params = {
                "permissions": permissions
            }
        resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", params=params, timeout=15)
        info = resp.json()
        pairs = []
        for s in info["symbols"]:
            if s["quoteAsset"] == "USDT" and (not is_trading or s["status"] == "TRADING"):
                pairs.append({
                    "symbol": s["symbol"],
                    "base": s["baseAsset"].upper(),
                    "status": s['status']
                })
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
        spot = {pair['base']: pair['symbol'] for pair in spot}
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

    def get_klines_window(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        all_klines = []
        start_ts = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
        current_ts = start_ts

        while current_ts < end_ts:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": current_ts,
                "endTime": end_ts,
                "limit": 1000,
            }
            resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15)
            if resp.status_code != 200 or not resp.json():
                break
            klines = resp.json()
            all_klines.extend(klines)
            current_ts = klines[-1][0] + 86400000  # next day
            time.sleep(0.1)

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)
        df = df[["date", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
        return df

    def probe_listing(self, symbol: str, start: str = "2021-01-01"):
        """
        Check whether a Binance symbol has kline data during the backtest window.
        'listed_date' means first available Binance daily kline on/after probe_start,
        not necessarily the true Binance listing date.
        """
        try:
            start_ts = int(
                datetime.datetime.strptime(start, "%Y-%m-%d")
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp() * 1000
            )

            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ts,
                "limit": 1000,
            }

            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=10
            )
            
            if resp.status_code != 200:
                return None

            klines = resp.json()
            if not klines:
                return None

            first_date = datetime.datetime.utcfromtimestamp(
                klines[0][0] / 1000
            ).strftime("%Y-%m-%d")

            # Fetch latest available candle directly
            latest_params = {
                "symbol": symbol,
                "interval": "1d",
                "limit": 1,
            }

            latest_resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=latest_params,
                timeout=10
            )

            if latest_resp.status_code == 200 and latest_resp.json():
                latest_kline = latest_resp.json()[-1]
                last_date = datetime.datetime.utcfromtimestamp(
                    latest_kline[0] / 1000
                ).strftime("%Y-%m-%d")
            else:
                last_date = datetime.datetime.utcfromtimestamp(
                    klines[-1][0] / 1000
                ).strftime("%Y-%m-%d")

            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

            days_since_last = (
                datetime.datetime.strptime(today, "%Y-%m-%d")
                - datetime.datetime.strptime(last_date, "%Y-%m-%d")
            ).days

            return {
                "backtest_start": start,
                "first_available_date_after_backtest_start": first_date,
                "available_at_backtest_start": first_date == start,
                "last_kline_date": last_date,
                "is_active": days_since_last <= 2,
                "delisted_date": last_date if days_since_last > 2 else None,
            }

        except Exception as e:
            print(f"Error probing {symbol}: {e}")
            return None