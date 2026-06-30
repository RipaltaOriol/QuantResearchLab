import pandas as pd
from engine.position import Position
from engine.cohort import Cohort

class BacktestEngine:
    def __init__(self, prices: pd.DataFrame, probings: dict = None, capital: float = 1):
        self.prices = prices
        self.probings = probings
        self.min_listed = 0
        self.signal: pd.Series | pd.DataFrame = None
        self.capital = capital

        # configurations
        self.holding_time = 5
        self.fee = 1 / 1_000
        self.slippage = 5 / 10_000
        self.margin = 0.0
        self.n_long_leg = 10
        self.n_short_leg = 10
        self.long_alloc = 0.5
        self.short_alloc = 0.5
        self.penalty = 500 / 10_000
        self.is_verbose = True


    def config(self, holding_time, min_listed):
        self.min_listed = min_listed
        self.holding_time = holding_time

    def verbose(self, flag: bool = True):
        self.is_verbose = flag

    def is_tradable(self, ticker: str, date):
        if self.probings is None:
            return True
        # FIX: verify these is not issues with this
        if ticker not in self.probings:
            return False
        info = self.probings[ticker]
        listed = info['listed']
        delisted = info['delisted']
        if date < listed or date > delisted:
            return False
        if (date - listed).days < self.min_listed:
            return False
        
        # TODO: in here you can add filterd (ie. min volume)
        return True


    def get_price(self, ticker: str, date) -> float | None
        if ticker not in self.prices.columns:
            return None
        price = self.prices.loc[date, ticker]
        return price if pd.notna(price) else None

    def open(self, ticker, direction, price, notional):
        if direction == "long":
            slippage = (1 + self.slippage)
            margin = 0
        else:
            slippage = (1 - self.slippage)
            margin = notional * self.margin
        entry = price * slippage
        fee = notional * self.fee
        shares = (notional - fee) / entry
        position = Position(ticker, shares, entry, direction, margin)
        return position, fee, margin + fee

    def close(self, position, price):
        # TODO: missing funding & margin
        if position.direction == "long":
            slippage = (1 - self.slippage)
            exit_p = price * slippage
            gross = position.shares * exit_p
            fee = gross * self.fee
        else:
            slippage = (1 + self.slippage)
            exit_p = price * slippage
            gross = position.shares * (position.entry - exit_p)
            fee = position.shares * exit_p * self.fee
        net = gross - fee
        return net, fee
    
    def open_cohort(self, curr_date, prev_date, capital):
        idx = self.prices.index.get_loc(curr_date)
        close_date = self.prices.index[idx + self.holding_time]
        cohort = Cohort(curr_date, close_date, capital)

        if prev_date not in self.signal.index: # no signal
            return cohort, 0.0
        
        signal = self.signal.loc[prev_date].dropna()
        tradable = signal[[t for t in signal.index if self.is_tradable(t, curr_date)]]

        if len(tradable) < self.n_long_leg + self.n_short_leg:
            return cohort, 0.0

        top = list(tradable.nlargest(self.n_long_leg).index)
        bot = [t for t in tradable.nsmallest(self.n_short_leg).index if t not in top]

        long_alloc = (capital * self.long_alloc) / max(len(top), 1)
        short_alloc = (capital * self.short_alloc) / max(len(bot), 1)
        used_cash = 0.0
        
        for ticker in top:
            price = self.get_price(ticker, curr_date)
            if not price: continue
            position, fee, _ = self.open(ticker, 'long', price, long_alloc)
            cohort.longs[ticker] = position
            cohort.fees += fee
            used_cash += long_alloc
        
        for ticker in bot:
            price = self.get_price(ticker, curr_date)
            if not price: continue
            position, fee, cost = self.open(ticker, 'short', price, short_alloc)
            cohort.shorts[ticker] = position
            cohort.fees += fee
            used_cash += cost
        
        return cohort, used_cash
    
    def close_cohort(self, cohort, date):
        for ticker in cohort.longs.keys():
            price = self.get_price(ticker, date)
            position = cohort.longs[ticker]
            if price:
                proceeds, fee = self.close(position, price)
                cash += proceeds
                cohort.fees += fee
                cohort.realized_pnl += position.pnl(price)
            del cohort.longs[ticker] # remove position
        for ticker in cohort.shorts.keys():
            price = self.get_price(ticker, date)
            position = cohort.shorts[ticker]
            if price:
                proceeds, fee = self.close(position, price)
                cash += proceeds
                cohort.fees += fee
                cohort.realized_pnl += position.pnl(price)
                # missing funding handles
            del cohort.shorts[ticker] # remove position
        cohort.closed = True
        return cash            

    def handle_delisting(self, cohort, date):
        cash = 0
        for ticker in cohort.longs.keys():
            delisted = self.probings[ticker].get('delisted', None)
            if delisted and date >= delisted:
                price = self.get_price(ticker, date)
                if price:
                    price = price * (1 - self.penalty)
                    proceeds, fee = self.close(cohort.longs[ticker], price)
                    cohort.fees += fee
                    cash += proceeds
                    if self.is_verbose:
                        print(f"DELIST long {ticker} on {date}")
                del cohort.longs[ticker]
            
        for ticker in cohort.shorts.keys():
            delisted = self.probings[ticker].get('delisted', None)
            if delisted and date >= delisted:
                price = self.get_price(ticker, date)
                if price:
                    price = price * (1 - self.penalty)
                    proceeds, fee = self.close(cohort.shorts[ticker], price)
                    cohort.fees += fee
                    cash += proceeds
                    if self.is_verbose:
                        print(f"DELIST short {ticker} on {date}")
                del cohort.shorts[ticker]
        return cash

    def handle_funding(self, cohort):
        pass

    def run(self, is_verbose = False):
        dates = sorted(self.signal.index)
        # TODO: trim dates to backtest window if needed

        price_fn = lambda ticker, date: self.get_price(ticker, date)
        capital = self.capital
        holdings, hist = [], []
        cache = []
        valuation = {}
        log = {}
        

        for idx, date in enumerate(dates):
            for cohort in holdings:
                capital += self.handle_delisting(cohort, date)
            
            # TODO: apply funding

            for cohort in holdings:
                if date >= cohort.close_date:
                    capital += self.close_cohort(cohort, date)
                    hist.append(cohort)
                    log_id = len(log)
                    log[f"{log_id} {date}"] = dict(
                        action = 'CLOSE',
                        date = date,
                        pnl = cohort.realized_pnl,
                        fess = cohort.fees
                    )
                else:
                    cache.append(cohort)
            holdings = cache

            nav = capital + sum(cohort.nav(price_fn) for cohort in holdings)
            cohort_alloc = nav / self.holding_time

            # TODO: be careful with this
            prev_date = dates[max(idx - 1, 0)]
            
            if cohort_alloc > 0:
                cohort, used_cash = self.open_cohort(date, prev_date, cohort_alloc)
                capital -= used_cash
                holdings.append(cohort)
                log[f"{log_id} {date}"] = dict(
                        action = 'OPEN',
                        date = date,
                        allocation = cohort_alloc,
                        longs = len(cohort.longs)
                        shorts = len(cohort.shorts)
                    )

            # FIX valuation
            valuation[date] = nav
            # FIX close anything still open