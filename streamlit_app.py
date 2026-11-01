from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Discovering Beta",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px;}
    h1 {color: #153b57; margin-bottom: .1rem;}
    .subtitle {font-size: 1.15rem; color: #415463; margin-bottom: 1rem;}
    .metric-card {background:#eaf5fa; border:1px solid #b7d9e8; border-radius:12px;
                  padding:11px 14px; min-height:88px;}
    .metric-title {font-size:.92rem; color:#50636f; font-weight:700;}
    .metric-value {font-size:1.8rem; color:#123f5b; font-weight:800;}
    .plain-language {background:#f7fbfd; border-left:5px solid #4b9cc5; border-radius:8px;
                     padding:9px 12px; margin:-4px 0 11px 0; font-size:.94rem;
                     line-height:1.45; color:#263942;}
    .question {background:#fff3c4; border-left:7px solid #e5aa00; border-radius:8px;
               padding:13px 16px; font-size:1.05rem;}
    div.stButton > button {font-size:1.05rem; font-weight:700; border-radius:9px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(data_version: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    _ = data_version  # Included in the cache key when the bundled data schema changes.
    prices = pd.read_parquet(ROOT / "adjusted_prices_2021_2025.parquet")
    prices.index = pd.to_datetime(prices.index)
    spy = pd.read_parquet(ROOT / "spy_adjusted_2021_2025.parquet")["SPY"]
    spy.index = pd.to_datetime(spy.index)
    universe = pd.read_csv(ROOT / "universe.csv")
    returns = prices.pct_change(fill_method=None)
    market = spy.pct_change(fill_method=None).rename("Market")
    universe = universe[universe["Ticker"].isin(returns.columns)].copy()
    universe["Company"] = universe["Security"] + "  [" + universe["Ticker"] + "]"
    return returns, market, universe.sort_values(["Security", "Ticker"])


returns, market, universe = load_data("gics-sectors-v1")
company_to_ticker = dict(zip(universe["Company"], universe["Ticker"]))


st.title("Discovering Beta")
st.markdown(
    '<div class="subtitle">Use real 2021–2025 daily stock returns to discover how strongly a company moves with the S&amp;P 500 market benchmark.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.9, 1], gap="large")
with left:
    selected_company = st.selectbox(
        "Search for a company",
        options=universe["Company"].tolist(),
        index=universe["Company"].tolist().index("Advanced Micro Devices  [AMD]")
        if "Advanced Micro Devices  [AMD]" in universe["Company"].tolist() else 0,
        help="Start typing a company name or ticker.",
    )
with right:
    prediction = st.radio(
        "Before revealing beta, predict:",
        ["Low beta (below 0.8)", "Average beta (around 1)", "High beta (above 1.2)"],
        horizontal=False,
    )

ticker = company_to_ticker[selected_company]
name = selected_company.rsplit("  [", 1)[0]
company_row = universe.loc[universe["Ticker"].eq(ticker)].iloc[0]
sector = company_row.get("GICS Sector", "")
sub_industry = company_row.get("GICS Sub-Industry", "")
paired = pd.concat([market, returns[ticker].rename("Stock")], axis=1).dropna()
x = paired["Market"].to_numpy() * 100
y = paired["Stock"].to_numpy() * 100
beta, alpha = np.polyfit(x, y, 1)
corr = float(np.corrcoef(x, y)[0, 1])
stock_sd = float(paired["Stock"].std(ddof=1) * np.sqrt(252) * 100)
market_sd = float(paired["Market"].std(ddof=1) * np.sqrt(252) * 100)

if st.session_state.get("last_ticker") != ticker:
    st.session_state.last_ticker = ticker
    st.session_state.revealed = False

b1, b2, _ = st.columns([1, 1, 3])
with b1:
    if st.button("Plot Returns", width="stretch", type="primary"):
        st.session_state.revealed = False
with b2:
    if st.button("Reveal Beta", width="stretch"):
        st.session_state.revealed = True

revealed = st.session_state.get("revealed", False)
category = "Low beta" if beta < 0.8 else "High beta" if beta > 1.2 else "Average beta"
guess = prediction.split(" (")[0]

chart_col, lesson_col = st.columns([2.35, 1], gap="large")

with chart_col:
    limit = max(4.0, float(np.nanpercentile(np.abs(np.r_[x, y]), 99.2)))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="markers", name="Daily paired returns",
        marker=dict(size=6, color="#3b8dc4", opacity=.32),
        hovertemplate="Market: %{x:.2f}%<br>Stock: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[-limit, limit], y=[-limit, limit], mode="lines",
        name="Reference slope = 1.0", line=dict(color="#93a2ab", width=2, dash="dash"),
    ))
    if revealed:
        fig.add_trace(go.Scatter(
            x=[-limit, limit], y=[alpha - beta * limit, alpha + beta * limit], mode="lines",
            name=f"Estimated beta = {beta:.2f}", line=dict(color="#d33f49", width=4),
        ))
    fig.update_layout(
        title=dict(text=f"{name} ({ticker}) vs. the S&P 500 Market Benchmark", x=.5, font=dict(size=22)),
        xaxis_title="Daily market return (%)",
        yaxis_title=f"Daily {ticker} return (%)",
        xaxis=dict(range=[-limit, limit], zeroline=True, zerolinewidth=1, zerolinecolor="#6d7a82"),
        yaxis=dict(range=[-limit, limit], zeroline=True, zerolinewidth=1, zerolinecolor="#6d7a82"),
        height=610, margin=dict(l=45, r=20, t=70, b=45),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        font=dict(size=14),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dce6ec")
    fig.update_yaxes(showgrid=True, gridcolor="#dce6ec")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        f"Real adjusted-price returns, 2021–2025 • {len(paired):,} paired trading days. "
        "Historical beta is an estimate, not a guarantee of future sensitivity."
    )

with lesson_col:
    if revealed:
        st.markdown(
            f'<div class="question"><b>Your prediction:</b> {guess}<br><b>Result:</b> {category}<br><br>'
            f'{"✅ Correct prediction!" if guess == category else "Compare your prediction with the regression slope."}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        beta_direction = "in the same direction as" if beta >= 0 else "in the opposite direction from"
        beta_size = abs(beta)
        if corr >= 0.6:
            together_words = "They often moved in the same direction."
        elif corr >= 0.3:
            together_words = "They sometimes moved in the same direction, but not consistently."
        elif corr > -0.3:
            together_words = "There was no clear, consistent pattern in how they moved together."
        elif corr > -0.6:
            together_words = "They sometimes moved in opposite directions, but not consistently."
        else:
            together_words = "They often moved in opposite directions."
        relative_risk = stock_sd / market_sd
        relative_words = "more" if stock_sd > market_sd else "less"
        sector_text = sector if isinstance(sector, str) and sector else "Not available"
        sub_industry_text = sub_industry if isinstance(sub_industry, str) and sub_industry else "Not available"

        metrics = [
            (
                "Beta (market sensitivity)", f"{beta:.2f}",
                f"When the market moves by 1%, {ticker} has historically moved about "
                f"{beta_size:.2f}% {beta_direction} the market, on average. "
                "This describes a historical relationship—not a prediction or an explanation of cause."
            ),
            (
                "Company industry", sector_text,
                f"GICS sector: {sector_text}. Sub-industry: {sub_industry_text}."
            ),
            (
                "Do the stock and market move together?", f"{corr:.2f}",
                f"{together_words} Use this simple scale: +1 means they move together, "
                "0 means no clear pattern, and −1 means they move in opposite directions. "
                "This score tells us how closely they move together—not how large the stock's moves are."
            ),
            (
                f"{ticker}'s total volatility", f"{stock_sd:.1f}%",
                "Higher % = larger price swings and more total risk. This is not an expected loss."
            ),
            (
                "Market's total volatility", f"{market_sd:.1f}%",
                f"For comparison, {ticker} was about {relative_risk:.1f}× as volatile as the market "
                f"({relative_words} total risk)."
            ),
        ]
        for title, value, explanation in metrics:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div></div>'
                f'<div class="plain-language">{explanation}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="question"><b>Look at the cloud before revealing the answer.</b><br><br>'
            'Would the best-fitting line be flatter than, similar to, or steeper than the gray slope-1 line?</div>',
            unsafe_allow_html=True,
        )
        st.info("Choose your prediction, then click **Reveal Beta**.")

st.divider()
st.subheader("What should we learn from the graph?")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Each dot is one trading day.**\n\nThe x-axis shows the market return; the y-axis shows the stock return.")
with c2:
    st.markdown("**The regression slope is beta.**\n\nA steeper line means greater sensitivity to broad market movements.")
with c3:
    st.markdown("**Beta is not total volatility.**\n\nStandard deviation includes both market risk and firm-specific risk.")

with st.expander("Method and data notes"):
    st.markdown(
        "- Daily adjusted closing prices, January 2021 through December 2025.\n"
        "- Market benchmark: SPY, used as a broad U.S. market proxy.\n"
        "- Beta is the OLS slope from stock daily return on SPY daily return.\n"
        "- Total volatility is the annualized standard deviation of daily returns (daily SD × √252).\n"
        "- Current-constituent data may be subject to survivorship bias.\n"
        "- The app is designed for classroom illustration, not investment advice."
    )

