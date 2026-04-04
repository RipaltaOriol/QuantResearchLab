import datetime
import streamlit as st

st.set_page_config(
    page_title="Mean-Reversion Research",
    page_icon="🌦️",
    layout="wide",
)

"""
# :material/query_stats: Mean-Reversion Research

"""

""  # Add a little vertical space. Same as st.write("").
""

cols = st.columns([1, 2, 2])

top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Selectbox for stock tickers
    tickers = st.multiselect(
        "Stock tickers",
        options=sorted(set(['STOCKS', "metals", 'gree', 'red'])),
        # default=st.session_state.tickers_input,
        placeholder="Choose stocks to compare. Example: NVDA",
        accept_new_options=True,
    )


bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)


horizon_map = {
    "1 Months": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}

with bottom_left_cell:
    div = st.columns([1, 1])
    fi = div[0]
    sec = div[1]
    with fi:
        d = st.date_input("When's your birthday", datetime.date(2019, 7, 6))
    with sec:
        end = st.date_input("When's your birthday", datetime.date(2019, 8, 6))

    timerange = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="6 Months",
    )


from numpy.random import default_rng as rng
import pandas as pd
df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])


with right_cell:
    st.line_chart(
    df,
    x="a",
    y=["b", "c"],
    color=["#FF0000", "#0000FF"],
    )
    # st.altair_chart(
    #     alt.Chart(
    #         normalized.reset_index().melt(
    #             id_vars=["Date"], var_name="Stock", value_name="Normalized price"
    #         )
    #     )
    #     .mark_line()
    #     .encode(
    #         alt.X("Date:T"),
    #         alt.Y("Normalized price:Q").scale(zero=False),
    #         alt.Color("Stock:N"),
    #     )
    #     .properties(height=400)
    # )
