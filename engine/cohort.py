class Cohort:
    def __init__(self, open_date, close_date, alloc):
        self.open_date = open_date
        self.close_date = close_date
        self.alloc = alloc
        self.longs = {}
        self.shorts = {}
        self.closed = False
        self.realized_pnl = 0.0
        self.fees = 0.0
        self.funding = 0.0
    
    def nav(self, price_fn):
        v = 0.0
        for ticker, pos in self.longs.items():
            p = price_fn(ticker)
            if p: v += pos.market_value(p)
        for ticker, pos in self.shorts.items():
            p = price_fn(ticker)
            if p: v += pos.margin + pos.pnl(p) + pos.funding
        return v