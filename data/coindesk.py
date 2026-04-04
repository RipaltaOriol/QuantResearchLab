import os
import requests
import datetime
import pandas as pd

from dotenv import load_dotenv



class CoinDesk:
    def __init__(self):
        pass

    def get_OHLC(self, instrument, limit = 365, to_ts = None, gropus = "OHLC", market = "cadli"):

        if to_ts == None:
            to_ts = datetime.datetime.now().timestamp()

        params = {
            "market": market,
            "instrument": instrument,
            "limit": limit,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping":"true",
            "response_format":"JSON",
            "to_ts": int(to_ts),
            "groups": gropus,
            "api_key":"e1ce231cb8d7f66893ae59043728c02a04d6569301f93c292174a6c65c45bddf"}

        response = requests.get('https://data-api.coindesk.com/index/cc/v1/historical/days',
            params = params,
            headers = {"Content-type":"application/json; charset=UTF-8"}
        )

        all = []

        payload = response.json()
        data = payload.get('Data', [])

        # if not data:
        #     break

        all.extend(data)

        df = pd.DataFrame(all)
        df['DATE'] = pd.to_datetime(df['TIMESTAMP'], unit = 's', utc = True)
        df = df.sort_values('TIMESTAMP')

        # if df['TIMESTAMP'].duplicated.any():
        #     print("WARNING: Duplicate timestamps found!")

        return df

    def get_historical(self, instrumetns, limit = 365, to_ts = None, groups = "OHLC", market = "cadli"):

        out = {}

        for instrument in instrumetns:

            out[instrument] = self.get_OHLC(instrument, limit, to_ts)

        return out
