class Position:
    def __init__(self, ticker, shares, entry, direction, margin: float = 0.0):
        self.ticker = ticker
        self.shares = shares
        self.entry = entry
        self.direction = direction
        self.margin = margin
        self.funding = 0.0

    def market_value(self, price):
        if self.direction == "long":
            return self.shares * price
        else:
            return 0.0

    def pnl(self, price):
        if self.direction == "long":
            return self.shares * (price - self.entry)
        else:
            return self.shares * (self.entry - price)
    
    def notinal(self, price):
        return self.shares * price
