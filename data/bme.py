import pandas as pd

def fetch():
    url = "https://en.wikipedia.org/wiki/IBEX_35"
    tables = pd.read_html(url)

    # Find components table
    df = None
    for t in tables:
        if "Ticker" in t.columns and "Company" in t.columns:
            df = t.copy()
            break

    if df is None:
        raise ValueError("IBEX table not found")

    df = df.rename(columns={
        "Ticker": "Symbol",
        "Company": "Security",
        "Sector": "Sector"
    })

    df["Symbol"] = df["Symbol"].str.replace(".MC", "", regex=False)

    print(df.head())
