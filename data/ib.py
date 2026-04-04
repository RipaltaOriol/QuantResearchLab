"""
Interactive Brokers data fetching module using the IBAPI.

Provides a thin wrapper around IBKRApp to request historical OHLCV data
for equities and return a cleaned pandas DataFrame.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Thread

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class IBKRApp(EWrapper, EClient):
    """Minimal IBKR API client that collects historical bar data."""

    def __init__(self):
        EClient.__init__(self, self)
        self.data: list[dict] = []
        self.is_finished: bool = False


    def historicalData(self, reqId: int, bar) -> None:
        self.data.append({
            "datetime": bar.date,
            "open":     bar.open,
            "high":     bar.high,
            "low":      bar.low,
            "close":    bar.close,
            "volume":   bar.volume,
        })

    def historicalDataEnd(self, reqId: int, _start: str, _end: str) -> None:
        self.is_finished = True


class IBAPI:

    @staticmethod
    def get_data(
        symbol: str | None = None,
        *,
        isin: str | None = None,
        host: str = "127.0.0.1",
        port: int = 7496,
        client_id: int = 123,
        currency: str = "EUR",
        exchange: str = "SMART",
        duration: str = "1 Y",
        bar_size: str = "1 day",
        csv_path: str | None = None,
        csv_filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Interactive Brokers.

        Exactly one of ``symbol`` or ``isin`` must be provided.

        :param symbol: Ticker symbol (e.g. "NVDA", "SAB1").
        :param isin: ISIN (e.g. "US0378331005"). When supplied, IB resolves the
            contract via its smart router; ``exchange`` must be "SMART".
        :param host: TWS / IB Gateway host.
        :param port: 7496 live, 7497 paper.
        :param client_id: Unique connection ID.
        :param currency: Contract currency ("USD", "EUR", ...). Required even
            when identifying by ISIN, as ISINs do not encode currency.
        :param exchange: Routing exchange, default "SMART". Must be "SMART"
            when using an ISIN.
        :param duration: IB duration string ("1 Y", "6 M", "30 D", ...).
        :param bar_size: IB bar size ("1 day", "1 hour", ...).
        :param csv_path: Directory to save CSV in. If set, a CSV is written.
        :param csv_filename: Override the auto-generated filename (optional).

        :return: pd.DataFrame with columns: datetime, open, high, low, close, volume, return.
        """
        if (symbol is None) == (isin is None):
            raise ValueError("Provide exactly one of 'symbol' or 'isin'.")

        app = IBKRApp()
        app.connect(host, port, client_id)

        thread = Thread(target=app.run, daemon=True)
        thread.start()
        time.sleep(1)  # allow connection handshake to complete

        contract = Contract()
        contract.secType  = "STK"
        contract.exchange = exchange
        contract.currency = currency

        if isin is not None:
            contract.symbol    = ""
            contract.secIdType = "ISIN"
            contract.secId     = isin
        else:
            contract.symbol = symbol

        app.reqHistoricalData(
            reqId          = 1,
            contract       = contract,
            endDateTime    = time.strftime("%Y%m%d %H:%M:%S"),
            durationStr    = duration,
            barSizeSetting = bar_size,
            whatToShow     = "TRADES",
            useRTH         = 1,
            formatDate     = 1,
            keepUpToDate   = 0,
            chartOptions   = [],
        )

        while not app.is_finished:
            time.sleep(0.5)

        app.disconnect()

        df = pd.DataFrame(app.data)
        df["return"] = df["close"].pct_change()
        df = df.dropna(subset=["return"])
        df = df[["datetime", "open", "high", "low", "close", "volume", "return"]]

        if csv_path is not None:
            identifier = isin if isin is not None else symbol
            filename = csv_filename or f"stock_data_{identifier}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
            df.to_csv(f"{csv_path.rstrip('/')}/{filename}", index=False)

        return df

    @staticmethod
    def get_multiple(
        symbols: list[str] | None = None,
        *,
        isins: list[str] | None = None,
        host: str = "127.0.0.1",
        port: int = 7496,
        base_client_id: int = 200,
        currency: str = "EUR",
        exchange: str = "SMART",
        duration: str = "1 Y",
        bar_size: str = "1 day",
        prices: bool = False,
        csv_path: str | None = None,
        csv_filename: str | None = None,
        max_workers: int = 1,
        request_delay: float = 15.0,
    ) -> pd.DataFrame:
        """
        Fetch returns for multiple instruments and combine into a single DataFrame.

        Exactly one of ``symbols`` or ``isins`` must be provided.

        :param symbols: List of ticker symbols (e.g. ["NVDA", "AAPL"]).
        :param isins: List of ISINs (e.g. ["US0378331005", "US5949181045"]).
            Column names in the result will be the ISINs. ``exchange`` must be
            "SMART" when using ISINs.
        :param host: TWS / IB Gateway host.
        :param port: 7496 live, 7497 paper.
        :param currency: Contract currency ("USD", "EUR", ...). Required even
            when identifying by ISIN, as ISINs do not encode currency.
        :param exchange: Routing exchange, default "SMART". Must be "SMART"
            when using ISINs.
        :param duration: IB duration string ("1 Y", "6 M", "30 D", ...).
        :param bar_size: IB bar size ("1 day", "1 hour", ...).
        :param base_client_id: Client IDs are assigned as base + index to avoid conflicts.
        :param prices: If True, return closing prices instead of returns.
        :param csv_path: If set, writes the combined DataFrame to a timestamped CSV.
        :param csv_filename: Override the auto-generated filename (optional).
        :param max_workers: Parallel connections. Keep at 1 unless you know IB allows it.
        :param request_delay: Seconds between requests to respect IB pacing limits.

        :return: pd.DataFrame with datetime index, identifiers as columns, and
            prices or returns as values. Missing dates for any instrument are filled with NaN.
        """
        if (symbols is None) == (isins is None):
            raise ValueError("Provide exactly one of 'symbols' or 'isins'.")

        use_isins = isins is not None
        identifiers = isins if use_isins else symbols

        series: dict[str, pd.Series] = {}
        errors: dict[str, str]       = {}

        target = 'prices' if prices else 'returns'

        def fetch(identifier: str, client_id: int) -> tuple[str, pd.Series]:
            kwargs = dict(isin=identifier) if use_isins else dict(symbol=identifier)
            df = IBAPI.get_data(
                **kwargs,
                host=host,
                port=port,
                client_id=client_id,
                currency=currency,
                exchange=exchange,
                duration=duration,
                bar_size=bar_size,
            )
            return identifier, df.set_index("datetime")[target]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i, identifier in enumerate(identifiers):
                if i > 0 and max_workers == 1:
                    time.sleep(request_delay)
                future = pool.submit(fetch, identifier, base_client_id + i)
                futures[future] = identifier

            for future in as_completed(futures):
                identifier = futures[future]
                try:
                    ident, s = future.result()
                    series[ident] = s
                    print(f"[IB] {ident}: {len(s)} bars fetched.")
                except Exception as exc:
                    errors[identifier] = str(exc)
                    print(f"[IB] {identifier}: failed — {exc}")

        if errors:
            print(f"[IB] {len(errors)} instrument(s) failed: {list(errors.keys())}")

        combined = pd.DataFrame(series)

        if csv_path is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = csv_filename or f'equity_{target}_{timestamp}'
            combined.to_csv(f"{csv_path.rstrip('/')}/{filename}.csv")

        return combined


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]

    if len(tickers) == 1:
        data = IBAPI.get_data(symbol=tickers[0], csv_path=".")
        print(data.tail())
    else:
        returns = IBAPI.get_multiple(symbols=tickers, csv_path=".")
        print(returns.tail())
