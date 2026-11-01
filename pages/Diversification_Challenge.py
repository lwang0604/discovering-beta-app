from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ANNUALIZATION = np.sqrt(252) * 100

# Two pre-screened teaching sequences made only from real stocks and real returns.
# Their prefix portfolios at N = 1, 3, 10, 30, and 100 have declining volatility.
GUIDED_PORTFOLIOS = [
    "HWKN,HNI,HAYW,IBM,PVH,FANG,LTC,BBY,TEL,WRLD,NMIH,AN,HTH,PDFS,JBTM,HRMY,AUB,ENTG,TOL,UE,TDC,AME,RDN,INSW,PTGX,FND,LMT,LDOS,BSY,FHB,KFY,FLG,TPL,QNST,CELH,PSX,GAP,AVNT,FOX,EHC,GOOGL,PKG,CHRD,MMS,TTMI,CWST,CNR,REGN,ETR,AGX,DNOW,GNL,META,EAT,KNSL,FRPT,AMP,ACIW,UCTT,AVA,FFBC,RNST,CRL,LGND,PLXS,VSAT,OC,EGP,CNO,AHCO,IR,TMO,REZI,LYB,HLI,GEN,PGR,ROKU,EVRG,CNC,STC,RTX,BWXT,SPHR,MARA,RRC,HUBB,RGLD,AVGO,CVBF,COHU,ON,BKH,CBU,CARG,ES,GGG,ADEA,TMDX,XOM".split(","),
    "LZB,PFG,FIVN,NMIH,HRL,PENN,PZZA,MA,ENOV,MANH,BOOT,SBRA,HOMB,EPAC,ZBH,BOX,WEC,AMSF,ENR,AROC,KMPR,SNDR,PFE,CRI,MRSH,NEO,HASI,ADUS,EIX,IVZ,CRM,ANF,LOPE,EBC,CRC,MTDR,EL,EME,ADM,CNX,WMS,WDAY,HAS,JOE,CRL,USB,OMCL,MEDP,ARR,EXEL,CB,DVN,SABR,OVV,SPG,CBU,UCB,MNST,DELL,CCS,TXRH,WAT,HIW,GIS,OKTA,ALLY,NEM,AAMI,PYPL,DD,NVDA,BLK,AFG,COLL,V,BFH,MKTX,ACM,MGEE,UCTT,GL,WSO,HZO,ECPG,ELS,CBOE,ALRM,HUBB,PBF,CMCSA,BCO,COHU,SBSI,MCK,FSLR,PRIM,EW,MGY,EPAM,TR".split(","),
]

st.set_page_config(
    page_title="Diversification Challenge", page_icon="🧺", layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    .block-container {padding-top:.45rem; padding-bottom:1rem; max-width:1450px;}
    .course-banner {font-size:.98rem; font-weight:800; letter-spacing:.04em; color:#ffffff;
                    background:#153b57; border-left:8px solid #8ecae0; border-radius:8px;
                    padding:6px 12px; margin:0 0 .35rem 0;}
    h1 {color:#153b57; margin:.1rem 0; font-size:2.1rem !important; line-height:1.12 !important;}
    h3 {font-size:1.5rem !important; line-height:1.18 !important;}
    .subtitle {font-size:.98rem; color:#415463; margin-bottom:.45rem;}
    .result-box {background:#fff3c4; border-left:7px solid #e5aa00; border-radius:9px;
                 padding:13px 16px; font-size:1.03rem; line-height:1.5;}
    .movement-card {background:#ffe66d; border:1px solid #dfbd24; border-radius:12px;
                    padding:14px 18px; margin:12px 0 16px 0; text-align:center;}
    .movement-title {font-size:1.05rem; font-weight:750; color:#263942;}
    .movement-value {font-size:2.35rem; font-weight:850; color:#153b57; line-height:1.15;}
    .movement-note {font-size:.94rem; color:#334851; margin-top:5px;}
    .selected-box {background:#eaf5fa; border:1px solid #b7d9e8; border-radius:10px;
                   padding:10px 14px; font-size:1rem;}
    .random-result {border-radius:8px; padding:7px 9px; margin:6px 0; font-size:.92rem;
                    background:#f7fafc; border:1px solid #d7e2e8;}
    .st-key-conclusion_reveal button {background:#dff3fb !important; color:#153b57 !important;
                    border:2px solid #1769aa !important; border-radius:12px !important;
                    min-height:6.6rem !important; padding:12px 18px !important;
                    font-size:1.08rem !important; line-height:1.45 !important;
                    white-space:normal !important; text-align:left !important;}
    .st-key-conclusion_reveal button:hover {background:#cdebf7 !important;
                    border-color:#0f568c !important;}
    div.stButton > button {font-size:.96rem; font-weight:700; border-radius:9px;
                           min-height:2.3rem; padding:.3rem .55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

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
if "portfolio_weights" not in st.session_state:
    st.session_state.portfolio_weights = {}
if "weight_version" not in st.session_state:
    st.session_state.weight_version = 0
if "custom_weight_result" not in st.session_state:
    st.session_state.custom_weight_result = None
if "random_portfolio_results" not in st.session_state:
    st.session_state.random_portfolio_results = {}
if "portfolio_selection_attempts" not in st.session_state:
    st.session_state.portfolio_selection_attempts = {size: 0 for size in [1, 3, 10, 30, 100]}
if "show_risk_boxes" not in st.session_state:
    st.session_state.show_risk_boxes = False

st.markdown('<div class="course-banner">FIN 331 &nbsp;|&nbsp; Towson University</div>', unsafe_allow_html=True)
st.title("Diversification Challenge")
st.markdown(
    '<div class="subtitle">Choose real companies and watch how each new stock changes portfolio risk using actual 2021–2025 daily returns.</div>',
    unsafe_allow_html=True,
)

with st.expander(
    "Build Your Own Portfolio (optional)",
    expanded=bool(st.session_state.portfolio_tickers),
):
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
    st.session_state.portfolio_weights = {}
    st.session_state.custom_weight_result = None
    st.session_state.weight_version += 1
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
            prior_weights = st.session_state.portfolio_weights
            if set(prior_weights) == set(previous):
                before_returns = before_frame.mul(pd.Series(prior_weights)).sum(axis=1)
            else:
                before_returns = before_frame.mean(axis=1)
            before_sd = annualized_sd(before_returns)
            paired = pd.concat([returns[ticker], before_returns.rename("Portfolio")], axis=1).dropna()
            correlation = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
        updated = previous + [ticker]
        after_returns = returns[updated].dropna().mean(axis=1)
        after_sd = annualized_sd(after_returns)
        selected.append(ticker)
        total = len(selected)
        st.session_state.portfolio_weights = {symbol: 1 / total for symbol in selected}
        st.session_state.custom_weight_result = None
        st.session_state.weight_version += 1
        st.session_state.portfolio_history.append((total, after_sd))
        st.session_state.last_comparison = {
            "ticker": ticker, "stock_sd": stock_sd, "before_sd": before_sd,
            "after_sd": after_sd, "correlation": correlation, "total": total,
        }
        st.rerun()

selected = st.session_state.portfolio_tickers
if selected:
    stock_word = "stock" if len(selected) == 1 else "stocks"
    displayed_tickers = ", ".join(selected[:12])
    if len(selected) > 12:
        displayed_tickers += f", and {len(selected) - 12} more"
    st.markdown(
        f'<div class="selected-box"><b>Current portfolio ({len(selected)} {stock_word}):</b> '
        + displayed_tickers + "</div>",
        unsafe_allow_html=True,
    )

    if len(selected) <= 30:
        with st.expander("Change Portfolio Weights", expanded=False):
            st.write("Enter each weight as a percentage. The total must equal 100%.")
            input_weights = {}
            cols = st.columns(min(4, len(selected)))
            for idx, symbol in enumerate(selected):
                with cols[idx % len(cols)]:
                    input_weights[symbol] = st.number_input(
                        f"{symbol} weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(st.session_state.portfolio_weights.get(symbol, 1 / len(selected)) * 100),
                        step=1.0,
                        key=f"weight_{st.session_state.weight_version}_{symbol}",
                    )
            total_weight = sum(input_weights.values())
            st.write(f"**Total weight: {total_weight:.1f}%**")
            if st.button("Apply Weights", type="primary", width="stretch"):
                if not np.isclose(total_weight, 100.0, atol=0.05):
                    st.error("Weights must add up to 100% before they can be applied.")
                else:
                    decimal_weights = {symbol: value / 100 for symbol, value in input_weights.items()}
                    weighted_returns = returns[selected].dropna().mul(pd.Series(decimal_weights)).sum(axis=1)
                    weighted_sd = annualized_sd(weighted_returns)
                    st.session_state.portfolio_weights = decimal_weights
                    st.session_state.custom_weight_result = {
                        "sd": weighted_sd,
                        "weights": input_weights.copy(),
                    }
                    if st.session_state.portfolio_history:
                        st.session_state.portfolio_history[-1] = (len(selected), weighted_sd)
                    st.rerun()
    else:
        st.caption("The weight editor is available for portfolios of up to 30 stocks.")

    custom_result = st.session_state.custom_weight_result
    if custom_result:
        weight_text = ", ".join(
            f"{symbol} {weight:.1f}%" for symbol, weight in custom_result["weights"].items()
        )
        st.success(
            f"Custom-weight portfolio risk: {custom_result['sd']:.2f}% annualized standard deviation. "
            f"Weights: {weight_text}."
        )
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
        movement_value = "N/A"
        movement_note = "The first stock has no existing portfolio for comparison."
    else:
        change = after_sd - before_sd
        direction = "fell" if change < 0 else "rose"
        summary = (
            f"Portfolio risk {direction} from <b>{before_sd:.2f}%</b> to <b>{after_sd:.2f}%</b> "
            f"— a change of <b>{abs(change):.2f} percentage points</b>."
        )
        movement_value = f"{comparison['correlation']:.2f}"
        movement_note = "+1 = move together • 0 = no clear pattern • −1 = move in opposite directions"
    st.markdown(
        f'<div class="result-box"><b>Total stocks: {comparison["total"]}</b><br>{summary}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="movement-card">'
        '<div class="movement-title">Correlation coefficient with existing portfolio</div>'
        f'<div class="movement-value">{movement_value}</div>'
        f'<div class="movement-note">{movement_note}</div></div>',
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

st.markdown("### Portfolio Size Experiment")
control_col, chart_col = st.columns([0.82, 4.18], gap="medium")
random_sizes = [1, 3, 10, 30, 100]
random_palette = {
    1: "#2878B5", 3: "#F4C542", 10: "#2CA25F", 30: "#D94343", 100: "#111111"
}
random_icons = {1: "🔵", 3: "🟡", 10: "🟢", 30: "🔴", 100: "⚫"}

with control_col:
    st.markdown("**Select a portfolio size**")
    st.caption("Select a real equally weighted portfolio at each size.")
    random_clicked = None
    for size in random_sizes:
        if st.button(
            f"{random_icons[size]}  Select {size}",
            key=f"random_side_{size}",
            width="stretch",
        ):
            random_clicked = size

    if random_clicked is not None:
        attempt = st.session_state.portfolio_selection_attempts.get(random_clicked, 0)
        if attempt < len(GUIDED_PORTFOLIOS):
            random_tickers = GUIDED_PORTFOLIOS[attempt][:random_clicked]
        else:
            eligible = returns.columns[returns.notna().sum() >= 1200]
            rng = np.random.default_rng()
            random_tickers = rng.choice(
                eligible.to_numpy(), size=random_clicked, replace=False
            ).tolist()
        random_returns = returns[random_tickers].dropna().mean(axis=1)
        random_sd = annualized_sd(random_returns)
        st.session_state.random_portfolio_results[random_clicked] = {
            "size": random_clicked,
            "sd": random_sd,
            "tickers": random_tickers,
        }
        st.session_state.portfolio_selection_attempts[random_clicked] = attempt + 1
        st.session_state.show_risk_boxes = False
        st.rerun()

    random_results = st.session_state.random_portfolio_results
    for size in random_sizes:
        if size in random_results:
            result = random_results[size]
            st.markdown(
                f'<div class="random-result"><span style="color:{random_palette[size]};'
                f'font-size:1.25rem">★</span> <b>{size} stocks:</b> '
                f'{result["sd"]:.2f}%</div>',
                unsafe_allow_html=True,
            )
    if random_results and st.button("Clear random results", width="stretch"):
        st.session_state.random_portfolio_results = {}
        st.session_state.show_risk_boxes = False
        st.rerun()

with chart_col:
    st.subheader("How Does Portfolio Risk Change as We Add More Stocks?")
    all_sizes_selected = all(size in random_results for size in random_sizes)
    if all_sizes_selected and not st.session_state.show_risk_boxes:
        if st.button(
            "What should we conclude?\n"
            "On average, portfolio risk declines as stocks are added, but at a decreasing rate.\n"
            "Click to reveal the two parts of total risk →",
            key="conclusion_reveal",
            width="stretch",
        ):
            st.session_state.show_risk_boxes = True
            st.rerun()
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
        line=dict(color="#1769AA", width=3), name="Average of 1,000 randomly selected portfolios",
    ))
    spy_vol = float(results["SPY_Volatility_Pct"].iloc[0])
    curve.add_hline(
        y=spy_vol, line=dict(color="#C43D3D", width=2, dash="dash"),
        annotation_text=f"S&P 500 benchmark: {spy_vol:.1f}%",
        annotation_position="bottom right",
    )

    label_positions = {
        1: "top right", 3: "top left", 10: "top right", 30: "top center", 100: "top center"
    }
    for size in random_sizes:
        if size in random_results:
            result = random_results[size]
            curve.add_trace(go.Scatter(
                x=[size], y=[result["sd"]], mode="markers+text",
                text=[f"N={size}"], textposition=label_positions[size],
                marker=dict(
                    symbol="star", size=20, color=random_palette[size],
                    line=dict(color="white", width=1.5),
                ),
                name=f"Random {size}-stock portfolio",
                hovertemplate=(
                    f"Random {size}-stock portfolio<br>Annualized SD: "
                    f"{result['sd']:.2f}%<extra></extra>"
                ),
                showlegend=False,
            ))

    history = st.session_state.portfolio_history
    if history:
        hx, hy = zip(*history)
        curve.add_trace(go.Scatter(
            x=hx, y=hy, mode="markers", marker=dict(
                symbol="diamond", size=11, color="#7A5AF8",
                line=dict(color="white", width=1),
            ), name="Student-built portfolio",
        ))

    if all_sizes_selected and st.session_state.show_risk_boxes:
        curve.add_annotation(
            xref="paper", yref="paper", x=.69, y=1.02, showarrow=False,
            text="<b>Portfolio's Total Risk = Diversifiable Risk + Market Risk</b>",
            font=dict(size=15, color="#153B57"),
        )
        curve.add_annotation(
            xref="paper", yref="paper", x=.69, y=.86, showarrow=False, align="left",
            text=(
                "<b>Diversifiable Risk</b><br>"
                "The risk eliminated by adding stocks.<br>"
                "It comes from random events unique to a particular firm.<br>"
                "Examples: lawsuits, strikes, R&amp;D outcomes, and major contracts."
            ),
            bgcolor="#DFF3FB", bordercolor="#1769AA", borderwidth=2,
            borderpad=15, font=dict(size=16, color="#153B57"),
        )
        curve.add_annotation(
            xref="paper", yref="paper", x=.69, y=.34, showarrow=False, align="left",
            text=(
                "<b>Market Risk</b><br>"
                "The risk that remains even if the portfolio holds every stock.<br>"
                "It comes from factors that systematically affect most firms.<br>"
                "Examples: war, inflation, recessions, and high interest rates.<br>"
                "<b>Market risk cannot be eliminated by diversification.</b>"
            ),
            bgcolor="#FFF0E8", bordercolor="#C43D3D", borderwidth=2,
            borderpad=15, font=dict(size=16, color="#6F2525"),
        )

    random_ymax = max(
        [65.0] + [result["sd"] * 1.15 for result in random_results.values()]
    )
    curve.update_layout(
        height=500, margin=dict(l=58, r=20, t=20, b=48),
        xaxis_title="Number of Stocks", yaxis_title="Portfolio Risk, σₚ (%)",
        xaxis=dict(
            type="log", range=[0, np.log10(550)],
            tickvals=[1, 3, 10, 30, 100, 300, 500], gridcolor="#edf2f5",
        ),
        yaxis=dict(range=[0, random_ymax], gridcolor="#dce6ec"),
        plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.08),
        font=dict(size=13),
    )
    st.plotly_chart(curve, width="stretch", config={"displaylogo": False})

st.caption(
    "Daily constant-weight portfolios using real adjusted-price returns from 2021–2025. "
    "New portfolios begin equally weighted; student-entered weights are rebalanced daily. "
    "Historical results are for teaching, not investment advice."
)
