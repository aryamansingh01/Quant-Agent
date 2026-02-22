#!/usr/bin/env python
# coding: utf-8
# ============================================================
# app.py — Quant Agent | AI Strategy Research Platform
# Run   : streamlit run app.py
# ============================================================

import json
import streamlit as st
import pandas    as pd
import numpy     as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_engine import (
    run_full_pipeline,
    generate_pdf_report,
    generate_html_report,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title            = "⚡ Quant Agent",
    page_icon             = "⚡",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f172a; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background-color: #1e293b; }

  .kpi-card {
    background    : #1e293b;
    border-radius : 12px;
    padding       : 18px 20px;
    border-left   : 4px solid #3b82f6;
    margin-bottom : 8px;
  }
  .kpi-value { font-size:30px; font-weight:700; color:#60a5fa; }
  .kpi-label { font-size:11px; color:#94a3b8; margin-top:3px; letter-spacing:.5px; }

  .section-header {
    font-size     : 18px;
    font-weight   : 700;
    color         : #60a5fa;
    border-bottom : 2px solid #1e40af;
    padding-bottom: 6px;
    margin        : 24px 0 14px 0;
  }

  .stButton > button {
    background    : linear-gradient(135deg, #2563eb, #1d4ed8);
    color         : white !important;
    border        : none;
    border-radius : 8px;
    font-weight   : 600;
    padding       : 10px 22px;
    transition    : all .2s ease;
    width         : 100%;
  }
  .stButton > button:hover { opacity:.85; transform:translateY(-1px); }

  .stDownloadButton > button {
    background    : linear-gradient(135deg, #059669, #047857) !important;
    color         : white !important;
    border-radius : 8px !important;
    font-weight   : 600 !important;
    width         : 100%;
  }

  .stTextArea textarea {
    background    : #1e293b !important;
    color         : #e2e8f0 !important;
    border        : 1px solid #334155 !important;
    border-radius : 10px !important;
    font-size     : 15px !important;
  }

  .stDataFrame  { border-radius:10px; overflow:hidden; }
  .stAlert      { border-radius:10px !important; }
  .stSuccess    { border-left:4px solid #10b981 !important; }

  .streamlit-expanderHeader {
    background:   #1e293b !important;
    border-radius:8px !important;
    color:        #60a5fa !important;
    font-weight:  600 !important;
  }

  div[data-testid="stSidebar"] .stButton > button {
    background  : #334155 !important;
    font-size   : 12px;
    font-weight : 500;
    text-align  : left;
    padding     : 8px 14px;
  }
  div[data-testid="stSidebar"] .stButton > button:hover {
    background  : #3b82f6 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Demo prompts ─────────────────────────────────────────────
DEMO_PROMPTS = [
    "60-day momentum on SPY",
    "mean reversion SP500",
    "vol targeting on QQQ",
    "20-day momentum on SP500 large caps with 10bps costs",
    "short-term mean reversion on QQQ with 5bps costs",
]

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 4px 0'>
      <span style='font-size:40px'>&#9889;</span><br>
      <span style='font-size:20px; font-weight:700; color:#60a5fa'>Quant Agent</span><br>
      <span style='font-size:11px; color:#64748b'>AI Strategy Research Platform</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**📋 Demo Strategies**")
    for dp in DEMO_PROMPTS:
        if st.button(dp, key=f"demo_{dp}", use_container_width=True):
            st.session_state["prompt_input"] = dp
            st.rerun()

    st.divider()
    st.markdown("**⚙️ Backtest Settings**")

    start_date = st.date_input(
        "Start Date",
        value     = pd.Timestamp("2015-01-01"),
        min_value = pd.Timestamp("2000-01-01"),
        max_value = pd.Timestamp("today"),
        key       = "start_date",
    )
    end_date = st.date_input(
        "End Date",
        value     = pd.Timestamp("today"),
        min_value = pd.Timestamp("2000-01-01"),
        max_value = pd.Timestamp("today"),
        key       = "end_date",
    )

    st.slider(
        "In-Sample Split %",
        min_value = 50, max_value = 90,
        value=70, step=5, key="is_split",
        help="% of data used for in-sample backtest",
    )

    st.divider()
    st.markdown("**📖 Signal Reference**")
    st.markdown("""
| Keyword | Type |
|---------|------|
| `momentum` | Trend-following |
| `mean reversion` | Contrarian |
| `vol targeting` | Risk-budget |

**Universes:** SPY · SP500 · QQQ · IWM · DIA
    """)
    st.divider()
    st.caption("Quant Agent v1.0 © 2026")
    st.caption("Data: FMP + FRED | Engine: NumPy/Pandas")

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:10px 0 6px 0'>
  <h1 style='margin:0; font-size:36px; color:#f1f5f9'>
    &#9889; Quant Agent
    <span style='font-size:14px; color:#64748b; font-weight:400;
                 margin-left:12px; vertical-align:middle'>
      AI-Powered Systematic Strategy Research
    </span>
  </h1>
  <p style='color:#94a3b8; font-size:13px; margin-top:4px'>
    Type any strategy idea in plain English — the agent parses,
    backtests, and delivers a full research report automatically.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Strategy input + run button ──────────────────────────────
col_input, col_btn = st.columns([5, 1])

with col_input:
    prompt = st.text_area(
        label            = "Strategy Idea",
        value            = st.session_state.get("prompt_input", DEMO_PROMPTS[0]),
        height           = 90,
        placeholder      = (
            "Examples:\n"
            "  • '60-day momentum on SPY'\n"
            "  • 'mean reversion SP500 with 5bps costs'\n"
            "  • 'vol targeting on QQQ'"
        ),
        key              = "prompt_input",
        label_visibility = "collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "🚀 Generate\nReport",
        use_container_width=True,
        key="run_btn",
    )

st.caption(
    "✅ Signals: **momentum** · **mean_reversion** · **vol_targeting**  "
    "| Universes: **SPY · SP500 · QQQ · IWM · DIA**  "
    "| Params: lookback · hold · bps costs"
)

# ── Session state init ───────────────────────────────────────
if "report"      not in st.session_state: st.session_state["report"]      = None
if "m"           not in st.session_state: st.session_state["m"]           = None
if "sp"          not in st.session_state: st.session_state["sp"]          = None
if "last_prompt" not in st.session_state: st.session_state["last_prompt"] = ""

# ── Pipeline trigger ─────────────────────────────────────────
if run_btn and prompt.strip():
    with st.spinner("🔄 Parsing → fetching data → backtesting → computing metrics..."):
        try:
            result = run_full_pipeline(
                prompt,
                start = str(start_date),
                end   = str(end_date),
            )
            st.session_state["report"]      = result
            st.session_state["m"]           = result["metrics"]
            st.session_state["sp"]          = result["spec"]
            st.session_state["last_prompt"] = prompt

        except ValueError as e:
            st.error(f"⚠️ Parse error: {e}")
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.info("💡 Engine auto-falls back to synthetic GBM data if FMP fails.")

# ── Welcome screen ───────────────────────────────────────────
if st.session_state["report"] is None:
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.info(
            "👆 **Enter a strategy idea** above and click **Generate Report**\n\n"
            "Or pick one of the **Demo Strategies** in the sidebar to get started instantly."
        )
    st.stop()

# ── Results ───────────────────────────────────────────────────
report = st.session_state["report"]
m      = st.session_state["m"]
sp     = st.session_state["sp"]

# ── Data source banner ───────────────────────────────────────
ds = report.get("data_sources", {})
price_src = ds.get("price_source", "unknown")
vix_src   = ds.get("vix_source",   "unknown")
ticker    = ds.get("fmp_ticker",   "?")
fmp_rows  = ds.get("fmp_rows",    0)
vix_rows  = ds.get("vix_rows",    0)

using_synthetic_price = price_src == "synthetic_gbm"
using_synthetic_vix   = vix_src   == "synthetic_constant"

if using_synthetic_price:
    st.error(
        "⚠️ **FMP API failed — price data is SYNTHETIC (GBM simulation), not real market data.**\n\n"
        "The results below are **illustrative only** and carry no predictive value.\n\n"
        "**How to fix:**\n"
        "- Check your FMP API key is valid at financialmodelingprep.com\n"
        "- Set it: `export FMP_API_KEY=your_key` then restart Streamlit\n"
        "- Or hardcode it in `quant_engine.py` → `FMP_API_KEY = 'your_key'`\n"
        "- Free tier supports: SPY, QQQ, IWM, DIA ✅ (SP500 constituent list needs paid plan)\n"
        f"- Ticker attempted: **{ticker}** | Rows returned: **{fmp_rows}**"
    )
elif fmp_rows > 0:
    st.success(
        f"✅ **Live FMP data loaded** | Ticker: `{ticker}` | Rows: `{fmp_rows}` | Real market prices"
    )

if using_synthetic_vix:
    st.warning(
        "⚠️ **FRED VIX data failed — regime analysis uses a flat VIX=20 constant.**\n\n"
        "VIX regime breakdown (Low/High VIX split) will show 50/50 and is not meaningful.\n\n"
        "**How to fix:**\n"
        "- FRED is free, no key needed — this is usually a network/timeout issue\n"
        "- Try again, or check if `fred.stlouisfed.org` is reachable from your machine"
    )
elif vix_rows > 0:
    st.success(f"✅ **Live FRED VIX loaded** | Rows: `{vix_rows}`")

# Success banner
st.success(
    f"✅ **{sp.get('signal','').replace('_',' ').title()}** on "
    f"**{sp.get('universe')}** | "
    f"Lookback: **{sp.get('lookback')}d** | "
    f"Hold: **{sp.get('hold')}d** | "
    f"Cost: **{sp.get('cost_bps')} bps** | "
    f"Generated: **{report.get('generated_at','')}**"
)

# =========================================================
# SECTION 1 — KPI CARDS + METRICS TABLE
# =========================================================
st.markdown('<div class="section-header">📊 Performance Metrics</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi(col, label, value, suffix="", color="#60a5fa"):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:{color}">{value}{suffix}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

kpi(c1, "SHARPE RATIO",  m["sharpe"],       color="#60a5fa")
kpi(c2, "MAX DRAWDOWN",  m["max_drawdown"], "%", "#f87171")
kpi(c3, "ANN. RETURN",   m["ann_return"],   "%", "#4ade80")
kpi(c4, "WIN RATE",      m["win_rate"],     "%", "#34d399")
kpi(c5, "SORTINO",       m["sortino"],      color="#a78bfa")
kpi(c6, "CALMAR",        m["calmar"],       color="#fb923c")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 📋 In-Sample vs Out-of-Sample")

is_days  = m["in_sample_days"]
oos_days = m["out_sample_days"]

table_df = pd.DataFrame({
    "Metric": [
        "Annualised Return", "Sharpe Ratio",  "Sortino Ratio",
        "Calmar Ratio",      "Max Drawdown",  "Ann. Volatility", "Win Rate",
    ],
    f"In-Sample ({is_days}d)": [
        f"{m['in_ann_return']}%", m["in_sharpe"],  m["in_sortino"],
        m["in_calmar"],           f"{m['in_max_drawdown']}%", "—", "—",
    ],
    f"Out-of-Sample ({oos_days}d)": [
        f"{m['out_ann_return']}%", m["out_sharpe"], m["out_sortino"],
        m["out_calmar"],           f"{m['out_max_drawdown']}%", "—", "—",
    ],
    "Full Period": [
        f"{m['ann_return']}%", m["sharpe"],  m["sortino"],
        m["calmar"],           f"{m['max_drawdown']}%",
        f"{m['ann_vol']}%",    f"{m['win_rate']}%",
    ],
})

def _colour_cell(val):
    try:
        v = float(str(val).replace("%", ""))
        if v > 0: return "color:#4ade80; font-weight:600"
        if v < 0: return "color:#f87171; font-weight:600"
    except Exception:
        pass
    return "color:#e2e8f0"

oos_col = f"Out-of-Sample ({oos_days}d)"
st.dataframe(
    table_df.style.map(_colour_cell, subset=[oos_col]),
    use_container_width=True,
    hide_index=True,
    height=285,
)

deg = round(m["in_sharpe"] - m["out_sharpe"], 3)

# Negative deg = OOS better than IS (unusual — likely period bias, not robustness)
# Positive deg = IS better than OOS (normal degradation — measure of overfit)
if deg < 0:
    icon  = "🟡"
    label = f"⚠️ OOS better than IS by {abs(deg)} — likely period bias, not robustness"
elif deg < 0.3:
    icon  = "🟢"
    label = "Robust ✅ — small degradation, strategy generalises well"
elif deg < 0.6:
    icon  = "🟡"
    label = "Moderate ⚠️ — some degradation, review parameter sensitivity"
else:
    icon  = "🔴"
    label = "High ❌ — large degradation, likely overfit to in-sample period"

st.caption(f"{icon} OOS Sharpe degradation (IS − OOS): **{deg}**  |  {label}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 🌦 VIX Regime Breakdown")
col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric("Low VIX Days  (<20)",   m["low_vix_days"])
col_r2.metric("Low VIX Sharpe",        m["low_vix_sharpe"])
col_r3.metric("High VIX Sharpe (≥20)", m["high_vix_sharpe"],
              delta=round(m["high_vix_sharpe"] - m["low_vix_sharpe"], 3))

# =========================================================
# SECTION 2 — EQUITY CURVE + DRAWDOWN
# =========================================================
st.markdown('<div class="section-header">📈 Equity Curve & Drawdown</div>',
            unsafe_allow_html=True)

eq  = report["equity_curve"]
dd  = report["drawdown_series"]
n   = len(eq)
spl = int(n * 0.70)

# Buy-and-hold benchmark: cumulative product of gross daily returns from backtest df
# backtest_df contains the raw returns so we can reconstruct B&H without extra API call
bt_df = report.get("backtest_df", None)
if bt_df is not None and "gross_ret" in bt_df.columns:
    # gross_ret = position * returns; B&H = always position=1, so use returns directly
    # net_returns col = strategy; we need raw market returns
    # gross_ret / position gives back daily return, but position can be 0
    # Safer: use net_ret + costs to get gross, then divide position where nonzero
    # Simplest correct approach: reconstruct from equity curve scaling
    raw_ret = bt_df["net_ret"] + bt_df["costs"]          # gross strategy return
    pos     = bt_df["position"].replace(0, float("nan")) # avoid div by zero
    mkt_ret = (raw_ret / pos).fillna(0)                  # market daily return
    bh_eq   = (1 + mkt_ret).cumprod()
    bh_idx_str = bh_eq.index.strftime("%Y-%m-%d").tolist()
    bh_vals    = bh_eq.values.tolist()
    show_bh    = True
else:
    show_bh = False

# Convert all timestamps to strings to avoid Plotly/Streamlit compat issues
eq_idx_str  = eq.index.strftime("%Y-%m-%d").tolist()
dd_idx_str  = dd.index.strftime("%Y-%m-%d").tolist()
split_date  = eq.index[spl].strftime("%Y-%m-%d")
min_dd_date = dd.idxmin().strftime("%Y-%m-%d")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.03,
    subplot_titles=(
        "Equity Curve  (base = $1.00)",
        "Drawdown  (%)",
    ),
)

# In-sample equity
fig.add_trace(go.Scatter(
    x=eq_idx_str[:spl], y=eq.iloc[:spl].tolist(),
    name=f"In-Sample ({m['in_sample_days']}d)",
    mode="lines", line=dict(color="#3b82f6", width=2),
    fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
), row=1, col=1)

# Out-of-sample equity
fig.add_trace(go.Scatter(
    x=eq_idx_str[spl:], y=eq.iloc[spl:].tolist(),
    name=f"Out-of-Sample ({m['out_sample_days']}d)",
    mode="lines", line=dict(color="#10b981", width=2.5),
    fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
), row=1, col=1)

# Buy-and-hold benchmark trace
if show_bh:
    fig.add_trace(go.Scatter(
        x=bh_idx_str, y=bh_vals,
        name="Buy & Hold (benchmark)",
        mode="lines",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    ), row=1, col=1)

# IS/OOS split line
fig.add_vline(
    x=split_date,
    line_dash="dash",
    line_color="#f59e0b",
    line_width=1.5,
)

# IS/OOS annotation (separate from vline to avoid crash)
fig.add_annotation(
    x=split_date,
    y=float(eq.max()) * 0.95,
    yref="y",
    text="IS / OOS",
    showarrow=False,
    xanchor="left",
    font=dict(color="#f59e0b", size=11, family="monospace"),
    bgcolor="rgba(30,41,59,0.85)",
    bordercolor="#f59e0b",
    borderwidth=1,
    row=1, col=1,
)

# $1 baseline
fig.add_hline(
    y=1.0, line_dash="dot",
    line_color="#475569", line_width=1,
    row=1, col=1,
)

# Drawdown area
fig.add_trace(go.Scatter(
    x=dd_idx_str, y=(dd.values * 100).tolist(),
    name="Drawdown %",
    mode="lines", line=dict(color="#f87171", width=1.5),
    fill="tozeroy", fillcolor="rgba(248,113,113,0.15)",
), row=2, col=1)

# Max DD annotation
fig.add_annotation(
    x=min_dd_date,
    y=float(dd.min() * 100),
    text=f"Max DD: {m['max_drawdown']}%",
    showarrow=True,
    arrowhead=2,
    arrowcolor="#f87171",
    font=dict(color="#f87171", size=11),
    ax=40, ay=-30,
    row=2, col=1,
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
    height=540, margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="#1e293b", bordercolor="#334155",
        borderwidth=1, orientation="h", x=0, y=1.08,
    ),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1e293b", font_color="#e2e8f0"),
)
fig.update_yaxes(gridcolor="#1e293b", row=2, col=1, ticksuffix="%")
fig.update_xaxes(showgrid=True, gridcolor="#1e293b", rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SECTION 3 — VIX REGIME HEATMAP
# =========================================================
st.markdown('<div class="section-header">🌡 VIX Regime Heatmap</div>',
            unsafe_allow_html=True)

vix = report.get("vix_series")
vix_valid = (
    vix is not None
    and isinstance(vix, pd.Series)
    and not vix.empty
)

if vix_valid:
    vix = vix.dropna()
    vix_idx_str = vix.index.strftime("%Y-%m-%d").tolist()

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=vix_idx_str, y=vix.values.tolist(),
        name="VIX", mode="lines",
        line=dict(color="#f59e0b", width=1.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
    ))

    hi_vals = np.where(vix.values >= 20, vix.values, np.nan).tolist()
    fig2.add_trace(go.Scatter(
        x=vix_idx_str, y=hi_vals,
        name="High VIX Zone (≥20)", mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.20)",
    ))

    fig2.add_hline(
        y=20, line_dash="dash", line_color="#3b82f6", line_width=1.2,
        annotation_text="Threshold: 20",
        annotation_font_color="#3b82f6",
        annotation_position="bottom right",
    )
    fig2.add_hline(
        y=30, line_dash="dash", line_color="#f87171", line_width=1.2,
        annotation_text="Elevated: 30",
        annotation_font_color="#f87171",
        annotation_position="bottom right",
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            bgcolor="#1e293b", bordercolor="#334155",
            borderwidth=1, orientation="h",
        ),
        hovermode="x unified", yaxis_title="VIX Level",
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    total  = m["low_vix_days"] + m["high_vix_days"]
    lo_pct = round(m["low_vix_days"] / total * 100, 1) if total > 0 else 0
    st.caption(
        f"📊 Low VIX: **{lo_pct}%** of days ({m['low_vix_days']}d)  |  "
        f"High VIX: **{round(100-lo_pct,1)}%** of days ({m['high_vix_days']}d)"
    )
else:
    st.info("ℹ️ VIX data not available — regime heatmap skipped.")

# =========================================================
# SECTION 4 — VIRTUAL P&L TABLE
# =========================================================
st.markdown('<div class="section-header">💹 Virtual P&L — Signal Activity</div>',
            unsafe_allow_html=True)

bt_df = report.get("backtest_df", pd.DataFrame())

if not bt_df.empty:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Trading Days",  m["total_days"])
    col_b.metric("In-Sample Days",      m["in_sample_days"])
    col_c.metric("Out-of-Sample Days",  m["out_sample_days"])
    col_d.metric("Capacity Est.",       f"${m['capacity_est_M']}M")

    st.markdown("<br>", unsafe_allow_html=True)

    display_df = (
        bt_df[["signal", "position", "gross_ret", "costs", "net_ret", "equity"]]
        .tail(30).copy()
    )
    display_df.index   = display_df.index.strftime("%Y-%m-%d")
    display_df.columns = [
        "Signal", "Position", "Gross Ret %", "Costs %", "Net Ret %", "Equity"
    ]
    for col in ["Gross Ret %", "Net Ret %", "Costs %"]:
        display_df[col] = (display_df[col] * 100).round(4).map("{:+.4f}%".format)
    display_df["Equity"]   = display_df["Equity"].map("{:.5f}".format)
    display_df["Signal"]   = display_df["Signal"].map("{:.4f}".format)
    display_df["Position"] = display_df["Position"].map("{:.4f}".format)

    def _pnl_colour(val):
        try:
            v = float(str(val).replace("%","").replace("+",""))
            if v > 0: return "color:#4ade80; font-weight:600"
            if v < 0: return "color:#f87171; font-weight:600"
        except Exception:
            pass
        return "color:#e2e8f0"

    st.dataframe(
        display_df.style
            .map(_pnl_colour, subset=["Net Ret %", "Gross Ret %"])
            .set_properties(**{"background-color":"#1e293b","color":"#e2e8f0"}),
        use_container_width=True,
        height=400,
    )
    st.caption(
        "📋 Showing last 30 trading days  |  "
        "Costs = one-way transaction cost deducted at each rebalance"
    )
else:
    st.warning("⚠️ Backtest DataFrame not available.")

# =========================================================
# SECTION 5 — STRATEGY THEORY + DEFINITIONS
# =========================================================
st.markdown('<div class="section-header">📚 Strategy Theory & Definitions</div>',
            unsafe_allow_html=True)

with st.expander("📖 Strategy Description", expanded=True):
    desc = report.get("strategy_description", "No description available.")
    st.markdown(desc)
    st.markdown(f"""
**Parameters used:**
- Universe: `{sp.get('universe')}`
- Signal: `{sp.get('signal')}`
- Lookback: `{sp.get('lookback')} days`
- Hold Period: `{sp.get('hold')} days`
- Transaction Cost: `{sp.get('cost_bps')} bps` one-way
    """)

with st.expander("📐 Metric Definitions"):
    st.markdown("""
| Metric | Formula | Hedge Fund Benchmark |
|--------|---------|----------------------|
| **Sharpe Ratio** | Ann. Return / Ann. Volatility | >1 acceptable · >2 excellent |
| **Sortino Ratio** | Ann. Return / Downside Std Dev | >1.5 good |
| **Calmar Ratio** | Ann. Return / \|Max Drawdown\| | >0.5 acceptable · >1 excellent |
| **Max Drawdown** | Worst peak-to-trough decline | <20% preferred |
| **Win Rate** | % days positive net P&L | 50–60% typical |
| **Ann. Volatility** | Daily Std × √252 | 10–20% typical equity |
| **Capacity Est.** | Final Equity × $100M | AUM before market impact |
    """)

with st.expander("🔬 In-Sample vs Out-of-Sample Methodology"):
    deg2 = round(m["in_sharpe"] - m["out_sharpe"], 3)
    if deg2 < 0:
        label2 = "🟡 OOS > IS — check for period bias (e.g. OOS caught a bull run)"
    elif deg2 < 0.3:
        label2 = "✅ Robust — small degradation, generalises well"
    elif deg2 < 0.6:
        label2 = "⚠️ Moderate — review parameter sensitivity"
    else:
        label2 = "❌ High — likely overfit to in-sample period"

    st.markdown(f"""
**Walk-forward split (zero look-ahead leakage)**
- **In-Sample ({m['in_sample_days']} days):** Strategy characterisation — parameters NOT tuned here
- **Out-of-Sample ({m['out_sample_days']} days):** Held-out live simulation — untouched until evaluation

**OOS Sharpe Degradation (IS − OOS): `{deg2}` → {label2}**

> Negative = OOS performed *better* than IS. This sounds good but is usually a period effect
> (e.g. IS included a choppy sideways market; OOS caught a sustained trend). It does **not**
> mean the strategy is robust — it means the two periods had different market regimes.

| Degradation (IS − OOS) | Interpretation |
|------------------------|----------------|
| < 0 | 🟡 OOS better than IS — check period bias |
| 0 – 0.30 | ✅ Generalises well |
| 0.30 – 0.60 | ⚠️ Mild overfit |
| > 0.60 | ❌ Likely overfit |

**Win Rate note:** A win rate below 50% is normal for momentum strategies.
Momentum profits come from asymmetric returns — wins are larger than losses on average,
not from winning more often. A 37% win rate with positive Sharpe is valid.
    """)

with st.expander("🌦 Regime Analysis Methodology"):
    st.markdown(f"""
**VIX-Based Regime Classification**

| Regime | VIX | Days | Sharpe |
|--------|-----|------|--------|
| Low VIX  | < 20  | {m['low_vix_days']}  | {m['low_vix_sharpe']} |
| High VIX | ≥ 20  | {m['high_vix_days']} | {m['high_vix_sharpe']} |

A robust strategy should show positive Sharpe in **both** regimes.
    """)

# =========================================================
# SECTION 6 — DOWNLOADS
# =========================================================
st.markdown('<div class="section-header">💾 Export Research Report</div>',
            unsafe_allow_html=True)

fname_base = (
    f"quant_agent_{sp.get('signal','strategy')}"
    f"_{sp.get('universe','SPY')}"
)
col_pdf, col_html, col_json = st.columns(3)

with col_pdf:
    st.markdown("**📄 PDF Report**")
    st.caption("Institutional research note with full tables")
    try:
        pdf_bytes = generate_pdf_report(report)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"{fname_base}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf",
        )
    except ImportError:
        st.warning("Run: `pip install reportlab`")
    except Exception as e:
        st.error(f"PDF error: {e}")

with col_html:
    st.markdown("**🌐 HTML Report**")
    st.caption("Self-contained — open in any browser")
    try:
        html_str = generate_html_report(report)
        st.download_button(
            label="📥 Download HTML",
            data=html_str.encode("utf-8"),
            file_name=f"{fname_base}.html",
            mime="text/html",
            use_container_width=True,
            key="dl_html",
        )
    except Exception as e:
        st.error(f"HTML error: {e}")

with col_json:
    st.markdown("**📦 JSON Metrics**")
    st.caption("Raw metrics for API or further analysis")
    try:
        safe_metrics = {
            k: v for k, v in m.items()
            if not isinstance(v, (pd.Series, pd.DataFrame))
        }
        safe_metrics["spec"]         = sp
        safe_metrics["generated_at"] = report.get("generated_at", "")
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(safe_metrics, indent=2).encode("utf-8"),
            file_name=f"{fname_base}_metrics.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
        )
    except Exception as e:
        st.error(f"JSON error: {e}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align:center; color:#475569; font-size:12px; padding:8px 0'>
  &#9889; <strong>Quant Agent v1.0</strong> &nbsp;|&nbsp;
  Systematic Strategy Research &nbsp;|&nbsp;
  Data: FMP + FRED &nbsp;|&nbsp;
  Stack: NumPy · Pandas · Plotly · ReportLab<br><br>
  <em>Past performance is not indicative of future results. Not investment advice.</em>
</div>
""", unsafe_allow_html=True)