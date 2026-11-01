from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ANNUALIZATION = np.sqrt(252) * 100

st.set_page_config(page_title="Diversification Challenge", page_icon="🧺", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1450px;}
    h1 {color:#153b57; margin-bottom:.1rem;}
    .subtitle {font-size:1.12rem; color:#415463; margin-bottom:1rem;}
    .result-box {background:#fff3c4; border-left:7px solid #e5aa00; border-radius:9px;
                 padding:13px 16px; font-size:1.03rem; line-height:1.5;}
    .selected-box {background:#eaf5fa; border:1px solid #b7d9e8; border-radius:10px;
                   padding:10px 14px; font-size:1rem;}
    div.stButton > button {font-size:1.03rem; font-weight:700; border-radius:9px;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Classroom Apps")
    st.page_link("streamlit_app.py", label="Discovering Beta", icon="📈")
    st.page_link("pages/Diversification_Challenge.py", label="Diversification Challenge", icon="🧺")


@st.cache_data
def load_data(version: str):
    _ = version
    prices = pd.read_parquet(ROOT / "adjusted_prices_2021_2025.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.where(prices > 0).ffill(limit=5)
    returns = prices.pct_change(fill_method=None)
    valid = returns.notna().sum()
    returns = returns.loc[:, valid[valid >= 1000].index]
    universe = pd.read_csv(ROOT / "universe.csv")
    universe = universe[universe["Ticker"].isin(returns.columns)].copy()
    universe["Company"] = universe["Security"] + "  [" + universe["Ticker"] + "]"
    universe = universe.sort_values(["Security", "Ticker"])
    results = pd.read_csv(ROOT / "Diversification_Results.csv")
    return returns, universe, results


def annualized_sd(series: pd.Series) -> float:
    return float(series.dropna().std(ddof=1) * ANNUALIZATION)


returns, universe, results = load_data("diversification-web-v1")
company_to_ticker = dict(zip(universe["Company"], universe["Ticker"]))

if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = []
if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = []
if "last_comparison" not in st.session_state:
    st.session_state.last_comparison = None

st.title("Diversification Challenge")
st.markdown(
    '<div class="subtitle">Choose real companies and watch how each new stock changes portfolio risk using actual 2021–2025 daily returns.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([3.2, 1], gap="large")
with left:
    chosen_company = st.selectbox(
        "Search for a company",
        universe["Company"].tolist(),
        help="Start typing a company name or ticker.",
    )
with right:
    st.write("")
    st.write("")
    add_clicked = st.button("＋ Add Company", type="primary", width="stretch")

clear_col, spacer = st.columns([1, 4])
with clear_col:
    clear_clicked = st.button("Clear Portfolio", width="stretch")

if clear_clicked:
    st.session_state.portfolio_tickers = []
    st.session_state.portfolio_history = []
    st.session_state.last_comparison = None
    st.rerun()

if add_clicked:
    ticker = company_to_ticker[chosen_company]
    selected = st.session_state.portfolio_tickers
    if ticker in selected:
        st.warning(f"{ticker} is already in the portfolio.")
    else:
        previous = selected.copy()
        stock_returns = returns[ticker].dropna()
        stock_sd = annualized_sd(stock_returns)
        before_sd = None
        correlation = None
        if previous:
            before_frame = returns[previous].dropna()
            before_returns = before_frame.mean(axis=1)
            before_sd = annualized_sd(before_returns)
            paired = pd.concat([returns[ticker], before_returns.rename("Portfolio")], axis=1).dropna()
            correlation = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
        updated = previous + [ticker]
        after_returns = returns[updated].dropna().mean(axis=1)
        after_sd = annualized_sd(after_returns)
        selected.append(ticker)
        total = len(selected)
        st.session_state.portfolio_history.append((total, after_sd))
        st.session_state.last_comparison = {
            "ticker": ticker, "stock_sd": stock_sd, "before_sd": before_sd,
            "after_sd": after_sd, "correlation": correlation, "total": total,
        }
        st.rerun()

selected = st.session_state.portfolio_tickers
if selected:
    st.markdown(
        f'<div class="selected-box"><b>Current portfolio ({len(selected)} stocks):</b> '
        + ", ".join(selected) + "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No companies selected yet. Add the first company to begin.")

comparison = st.session_state.last_comparison
if comparison:
    st.subheader(f"What happened when we added {comparison['ticker']}?")
    ticker = comparison["ticker"]
    before_sd = comparison["before_sd"]
    after_sd = comparison["after_sd"]
    if before_sd is None:
        summary = (
            f"{ticker} is the first stock. The portfolio's annualized standard deviation is "
            f"<b>{after_sd:.2f}%</b>."
        )
        movement = "Not available for the first stock."
    else:
        change = after_sd - before_sd
        direction = "fell" if change < 0 else "rose"
        summary = (
            f"Portfolio risk {direction} from <b>{before_sd:.2f}%</b> to <b>{after_sd:.2f}%</b> "
            f"— a change of <b>{abs(change):.2f} percentage points</b>."
        )
        movement = (
            f"Together-movement score with the existing portfolio: <b>{comparison['correlation']:.2f}</b>. "
            "+1 means they move together, 0 means no clear pattern, and −1 means opposite directions."
        )
    st.markdown(
        f'<div class="result-box"><b>Total stocks: {comparison["total"]}</b><br>{summary}<br>{movement}</div>',
        unsafe_allow_html=True,
    )

    labels = [f"{ticker}<br>alone"]
    values = [comparison["stock_sd"]]
    colors = ["#F0A202"]
    if before_sd is not None:
        labels.append(f"Existing portfolio<br>N = {comparison['total'] - 1}")
        values.append(before_sd)
        colors.append("#7A9CC6")
    labels.append(f"New portfolio<br>N = {comparison['total']}")
    values.append(after_sd)
    colors.append("#1769AA")
    bar = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.2f}%" for v in values], textposition="outside",
    ))
    bar.update_layout(
        height=360, margin=dict(l=45, r=20, t=25, b=45),
        yaxis_title="Annualized Standard Deviation (%)",
        yaxis=dict(range=[0, max(values) * 1.25], gridcolor="#dce6ec"),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        font=dict(size=14),
    )
    st.plotly_chart(bar, width="stretch", config={"displaylogo": False})

st.subheader("How Does Portfolio Risk Change as We Add More Stocks?")
x = results["N"]
curve = go.Figure()
curve.add_trace(go.Scatter(
    x=pd.concat([x, x.iloc[::-1]]),
    y=pd.concat([results["P95_Volatility_Pct"], results["P05_Volatility_Pct"].iloc[::-1]]),
    fill="toself", fillcolor="rgba(169,216,245,.55)", line=dict(color="rgba(0,0,0,0)"),
    name="5th–95th percentile", hoverinfo="skip",
))
curve.add_trace(go.Scatter(
    x=x, y=results["Average_Volatility_Pct"], mode="lines+markers",
    line=dict(color="#1769AA", width=3), name="Average of 1,000 random portfolios",
))
spy_vol = float(results["SPY_Volatility_Pct"].iloc[0])
curve.add_hline(
    y=spy_vol, line=dict(color="#C43D3D", width=2, dash="dash"),
    annotation_text=f"S&P 500 market benchmark: {spy_vol:.1f}%",
    annotation_position="bottom right",
)
history = st.session_state.portfolio_history
if history:
    hx, hy = zip(*history)
    curve.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers+text", text=[f"N={n}" for n in hx],
        textposition="top center", marker=dict(symbol="star", size=15, color="#F0A202",
        line=dict(color="#8A5700", width=1)), name="Your portfolio after each addition",
    ))
curve.update_layout(
    height=520, margin=dict(l=55, r=25, t=20, b=50),
    xaxis_title="Number of Stocks", yaxis_title="Annualized Portfolio Standard Deviation (%)",
    xaxis=dict(range=[0, 510], tickvals=[1, 30, 100, 300, 500], gridcolor="#edf2f5"),
    yaxis=dict(range=[0, 65], gridcolor="#dce6ec"),
    plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.08),
    font=dict(size=14),
)
st.plotly_chart(curve, width="stretch", config={"displaylogo": False})

st.caption(
    "Daily constant-weight portfolios using real adjusted-price returns from 2021–2025. "
    "Each selected portfolio is equally weighted and rebalanced daily. Historical results are for teaching, not investment advice."
)
