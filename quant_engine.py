#!/usr/bin/env python
# coding: utf-8
# ============================================================
# quant_engine.py — Quant Agent | Core Strategy Engine
# ============================================================
# Functions:
#   parse_english()       → English prompt to strategy spec dict
#   fetch_data()          → OHLCV + VIX via FMP + FRED (GBM fallback)
#   build_signal()        → momentum / mean_reversion / vol_targeting
#   backtest()            → vectorised P&L with hold-period & costs
#   metrics()             → Sharpe, Sortino, Calmar, IS/OOS, regimes
#   generate_report()     → bundle all artefacts into one dict
#   generate_pdf_report() → ReportLab PDF bytes
#   generate_html_report()→ self-contained HTML string
#   run_full_pipeline()   → one-call: prompt → full report dict
# ============================================================

import io, os, re, json, logging, warnings
import numpy  as np
import pandas as pd
import requests
from datetime import datetime
from typing   import Dict, Optional
from dotenv import load_dotenv
load_dotenv ()

# ── API Keys ─────────────────────────────────────────────────
FMP_API_KEY  = os.getenv("FMP_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE= os.getenv("FRED_BASE")
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)





DEFAULT_START        = "2015-01-01"
DEFAULT_END          = datetime.today().strftime("%Y-%m-%d")
TRANSACTION_COST_BPS = 10

# ============================================================
# SECTION 1 — parse_english : English prompt → strategy spec
# ============================================================

EXAMPLE_MAPPINGS: Dict[str, Dict] = {
    "20-day momentum on sp500 large caps with 10bps costs": {
        "universe": "SP500", "signal": "momentum",
        "lookback": 20, "hold": 5, "cost_bps": 10,
    },
    "60-day momentum on spy": {
        "universe": "SPY", "signal": "momentum",
        "lookback": 60, "hold": 10, "cost_bps": 10,
    },
    "mean reversion sp500": {
        "universe": "SP500", "signal": "mean_reversion",
        "lookback": 20, "hold": 5, "cost_bps": 10,
    },
    "vol targeting on qqq": {
        "universe": "QQQ", "signal": "vol_targeting",
        "lookback": 20, "hold": 5, "cost_bps": 10, "target_vol": 0.15,
    },
    "short-term mean reversion on qqq with 5bps costs": {
        "universe": "QQQ", "signal": "mean_reversion",
        "lookback": 10, "hold": 3, "cost_bps": 5,
    },
}


def parse_english(prompt: str) -> Dict:
    """
    Natural-language → structured spec dict.

    Pipeline:
      1. Exact match against EXAMPLE_MAPPINGS
      2. Fuzzy / keyword overlap (first 3 tokens)
      3. Regex extraction fallback

    Returns dict: universe, signal, lookback, hold, cost_bps
    """
    p = prompt.strip().lower()

    # 1. Exact match
    if p in EXAMPLE_MAPPINGS:
        logger.info("Exact spec match.")
        return EXAMPLE_MAPPINGS[p].copy()

    # 2. Fuzzy match on first 3 tokens
    for key, spec in EXAMPLE_MAPPINGS.items():
        anchor = key.split()[:3]
        if sum(w in p for w in anchor) >= 2:
            logger.info(f"Fuzzy match → '{key}'")
            return spec.copy()

    # 3. Regex extraction
    spec: Dict = {
        "universe": "SPY", "signal": "momentum",
        "lookback": 20, "hold": 5, "cost_bps": 10,
    }

    # Universe
    if   "sp500" in p or "s&p 500" in p or "large cap" in p: spec["universe"] = "SP500"
    elif "qqq"   in p or "nasdaq"  in p:                      spec["universe"] = "QQQ"
    elif "iwm"   in p or "russell" in p:                      spec["universe"] = "IWM"
    elif "dia"   in p or "dow"     in p:                      spec["universe"] = "DIA"
    elif "spy"   in p:                                         spec["universe"] = "SPY"

    # Signal type
    if   "mean reversion" in p or "reversion" in p:
        spec["signal"] = "mean_reversion"
    elif "vol target" in p or "vol-target" in p or "vol targeting" in p:
        spec["signal"]     = "vol_targeting"
        spec["target_vol"] = 0.15

    # Lookback
    m = re.search(r"(\d+)[\s\-]?day", p)
    if m: spec["lookback"] = int(m.group(1))

    # Hold period
    m = re.search(r"hold\s*(\d+)|(\d+)\s*day\s*hold", p)
    if m: spec["hold"] = int(m.group(1) or m.group(2))

    # Cost bps
    m = re.search(r"(\d+)\s*bps|(\d+)\s*basis", p)
    if m: spec["cost_bps"] = int(m.group(1) or m.group(2))

    logger.info(f"Parsed spec: {spec}")
    return spec


# ============================================================
# SECTION 2 — Data Helpers (FMP v3 + FRED)
# ============================================================

def _fmp_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily OHLCV from FMP stable endpoint:
      GET /stable/historical-price-eod/full?symbol=SYMBOL&from=...&to=...

    The old /api/v3/historical-price-full/ endpoint is now 403 for all
    accounts created after Aug 31 2025. The stable endpoint is the replacement.

    Returns empty DataFrame on any failure (caller falls back to GBM).
    """
    # New stable endpoint — symbol goes in query params, not the path
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    params = {
        "symbol": symbol,
        "from":   start,
        "to":     end,
        "apikey": FMP_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()

        # Stable endpoint returns a plain list of records
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            # Fallback: some responses wrap in a key
            records = payload.get("historical", payload.get("data", []))
        else:
            logger.warning(f"FMP: unexpected response type: {type(payload)}")
            return pd.DataFrame()

        if not records:
            logger.warning(f"FMP: empty records for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.columns = [c.lower() for c in df.columns]

        # Normalise date column
        if "date" not in df.columns:
            date_col = next(
                (c for c in df.columns if "date" in c or "time" in c), None
            )
            if date_col:
                df = df.rename(columns={date_col: "date"})
            else:
                logger.warning(f"FMP: no date column. Columns: {df.columns.tolist()}")
                return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Select OHLCV columns defensively
        ohlcv_map = {
            "open":   ["open"],
            "high":   ["high"],
            "low":    ["low"],
            "close":  ["adjclose", "close"],
            "volume": ["volume"],
        }
        col_select = {}
        available  = df.columns.tolist()
        for target, candidates in ohlcv_map.items():
            found = next((c for c in candidates if c in available), None)
            if found:
                col_select[found] = target

        df = df[list(col_select.keys())].rename(columns=col_select).astype(float)
        logger.info(f"FMP: fetched {len(df)} rows for {symbol}")
        return df

    except requests.HTTPError as e:
        logger.error(f"FMP HTTP error ({symbol}): {e.response.status_code} — {e.response.text[:200]}")
        return pd.DataFrame()
    except requests.RequestException as e:
        logger.error(f"FMP request failed ({symbol}): {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"FMP parse error ({symbol}): {e}")
        return pd.DataFrame()


def _fetch_vix(start: str, end: str) -> pd.Series:
    """
    Pull VIXCLS daily series from FRED public CSV endpoint.
    Returns empty Series on failure (caller synthesises VIX).
    """
    try:
        r = requests.get(
            FRED_BASE,
            params={"id": "VIXCLS"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]

        # FRED renamed DATE → OBSERVATION_DATE in 2024 — handle both
        if "OBSERVATION_DATE" in df.columns:
            df = df.rename(columns={"OBSERVATION_DATE": "DATE"})

        if "DATE" not in df.columns:
            logger.warning(f"FRED: no DATE column. Got: {df.columns.tolist()}")
            return pd.Series(dtype=float)

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).set_index("DATE")
        df.index.name = "date"

        vix_col = "VIXCLS" if "VIXCLS" in df.columns else df.columns[0]
        vix = df[vix_col].replace(".", np.nan)
        vix = pd.to_numeric(vix, errors="coerce").dropna().sort_index()
        vix = vix.loc[start:end]

        logger.info(f"FRED VIX: {len(vix)} rows fetched")
        return vix

    except Exception as e:
        logger.warning(f"FRED VIX fetch failed: {e}. Will use synthetic VIX.")
        return pd.Series(dtype=float)


def _synthetic_ohlcv(start: str, end: str) -> pd.DataFrame:
    """
    GBM-simulated OHLCV fallback (SPY-like: μ=10%, σ=18% ann.).
    Fixed seed for reproducibility across runs.
    """
    dates = pd.bdate_range(start=start, end=end)
    np.random.seed(42)
    mu, sigma, dt = 0.10, 0.18, 1 / 252
    log_ret = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * np.random.randn(len(dates))
    )
    px  = 200 * np.exp(np.cumsum(log_ret))
    rng = np.random.default_rng(0)
    df  = pd.DataFrame({
        "open":   px * (1 + 0.002 * rng.standard_normal(len(dates))),
        "high":   px * (1 + np.abs(0.005 * rng.standard_normal(len(dates)))),
        "low":    px * (1 - np.abs(0.005 * rng.standard_normal(len(dates)))),
        "close":  px,
        "volume": (1e7 * np.abs(rng.standard_normal(len(dates)) + 5)).astype(int),
    }, index=dates)
    df.index.name = "date"
    logger.info(f"Synthetic GBM data generated: {len(df)} rows")
    return df


# ============================================================
# SECTION 3 — fetch_data : unified data entry point
# ============================================================

PROXY_MAP = {
    "SP500": "SPY", "SPY": "SPY",
    "QQQ":   "QQQ", "IWM": "IWM", "DIA": "DIA",
}


def fetch_data(
    spec:  Dict,
    start: str = DEFAULT_START,
    end:   str = DEFAULT_END,
) -> tuple:
    """
    Fetch OHLCV + VIX for the universe defined in spec.

    Priority:
      1. FMP /api/v3  (real market data)
      2. Synthetic GBM (fallback when FMP fails / key missing)

    Returns:
        (DataFrame, data_sources: dict)

        data_sources keys:
            price_source : "fmp" | "synthetic_gbm"
            vix_source   : "fred" | "synthetic_constant"
            fmp_ticker   : str  — the ETF ticker attempted
            fmp_rows     : int  — rows returned (0 if failed)
            vix_rows     : int  — rows returned (0 if failed)
    """
    universe   = spec.get("universe", "SPY")
    etf_ticker = PROXY_MAP.get(universe, "SPY")

    logger.info(f"Fetching {etf_ticker} [{start} → {end}]")

    # ── Price data ───────────────────────────────────────────
    raw_df = _fmp_ohlcv(etf_ticker, start, end)

    if not raw_df.empty:
        df           = raw_df
        price_source = "fmp"
        fmp_rows     = len(df)
    else:
        logger.warning("FMP unavailable — falling back to synthetic GBM data.")
        df           = _synthetic_ohlcv(start, end)
        price_source = "synthetic_gbm"
        fmp_rows     = 0

    df["returns"] = df["close"].pct_change()

    # ── VIX data ─────────────────────────────────────────────
    vix = _fetch_vix(start, end)
    if not vix.empty:
        df = df.join(vix.rename("VIX"), how="left")
        df["VIX"] = df["VIX"].ffill().fillna(20.0)
        vix_source = "fred"
        vix_rows   = len(vix)
    else:
        # Flat constant fallback — avoids the abs() bias of the old random walk
        # which was pushing 80% of days into High-VIX territory incorrectly
        df["VIX"]  = 20.0
        vix_source = "synthetic_constant"
        vix_rows   = 0

    df.dropna(subset=["returns"], inplace=True)

    data_sources = {
        "price_source": price_source,
        "vix_source":   vix_source,
        "fmp_ticker":   etf_ticker,
        "fmp_rows":     fmp_rows,
        "vix_rows":     vix_rows,
    }

    logger.info(
        f"Data ready: {len(df)} rows | "
        f"{df.index.min().date()} → {df.index.max().date()} | "
        f"price={price_source} vix={vix_source}"
    )
    return df, data_sources


# ============================================================
# SECTION 4 — build_signal : price data → daily position signal
# ============================================================

def build_signal(data: pd.DataFrame, spec: Dict) -> pd.Series:
    """
    Generate a daily position signal in [-1, 1].

    Strategy types:
      momentum       → binary 1/0: long when close > SMA, flat otherwise.
                       Previously used (close-SMA)/SMA which gives ~0.02 for
                       SPY (price rarely deviates >2% from its own MA), causing
                       near-zero positions and flat equity curves on real data.
                       Binary is cleaner, fully-invested when signal is on.

      mean_reversion → z-score scaled: position = -z / z_scale, clipped [-1,1].
                       z_scale=2 means a 2-sigma move triggers a full position.
                       Negative z (oversold) → long; positive z (overbought) → short.

      vol_targeting  → position = target_vol / realised_vol, capped at 1.5×.
                       Scales down in high-vol (VIX spike) environments,
                       scales up in calm markets. Long-only, max 1.5× leverage.
    """
    stype    = spec.get("signal",   "momentum")
    lookback = spec.get("lookback", 20)
    close    = data["close"]

    if stype == "momentum":
        # Binary: fully invested (1.0) when price is above rolling SMA,
        # flat (0.0) when below. Clean, interpretable, realistic positions.
        sma    = close.rolling(lookback).mean()
        signal = (close > sma).astype(float)

    elif stype == "mean_reversion":
        # Z-score mean reversion: full short at +2σ, full long at -2σ
        sma    = close.rolling(lookback).mean()
        std    = close.rolling(lookback).std().replace(0, np.nan)
        z      = (close - sma) / std
        signal = (-z / 2).clip(-1, 1)   # z_scale=2 → ±1σ = half position

    elif stype == "vol_targeting":
        # Scale position inversely to realised vol to hit target annualised vol
        target = spec.get("target_vol", 0.15)
        rv     = (
            close.pct_change()
                 .rolling(lookback)
                 .std()
                 .replace(0, np.nan)
            * np.sqrt(252)
        )
        signal = (target / rv).clip(0, 1.5)

    else:
        raise ValueError(
            f"Unknown signal type: '{stype}'. "
            "Choose from: momentum, mean_reversion, vol_targeting"
        )

    signal.name = "signal"
    signal      = signal.fillna(0)

    active_pct = (signal != 0).mean() * 100
    mean_pos   = signal[signal != 0].mean() if (signal != 0).any() else 0
    logger.info(
        f"Signal built | type={stype} | lookback={lookback} | "
        f"active={active_pct:.1f}% of days | mean_position={mean_pos:.3f}"
    )
    return signal


# ============================================================
# SECTION 5 — backtest : vectorised P&L engine with costs
# ============================================================

def backtest(
    data:     pd.DataFrame,
    signal:   pd.Series,
    cost_bps: Optional[int] = None,
    spec:     Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Vectorised backtest with transaction cost deduction.

    Mechanics:
      • Rebalances every `hold` business days
      • One-way cost charged on |Δposition| at each rebalance
      • P&L uses prior-day position (no look-ahead bias)

    Returns DataFrame: signal, position, gross_ret, costs, net_ret, equity, vix
    """
    if cost_bps is None:
        cost_bps = (spec or {}).get("cost_bps", TRANSACTION_COST_BPS)

    cost_per_unit = cost_bps / 10_000
    hold          = (spec or {}).get("hold", 5)

    signal = signal.reindex(data.index).ffill().fillna(0)

    # Hold-period grid: hold signal constant for `hold` days
    position = pd.Series(0.0, index=data.index)
    for i in range(0, len(signal), hold):
        position.iloc[i : i + hold] = signal.iloc[i]

    # Transaction costs: |Δposition| × cost_per_unit
    daily_costs        = position.diff().abs() * cost_per_unit
    daily_costs.iloc[0] = position.iloc[0] * cost_per_unit  # entry cost day 1

    # Gross P&L (shift position 1 day — no look-ahead)
    gross_ret = position.shift(1) * data["returns"]
    net_ret   = gross_ret - daily_costs
    equity    = (1 + net_ret.fillna(0)).cumprod()

    result = pd.DataFrame({
        "signal":    signal,
        "position":  position,
        "gross_ret": gross_ret,
        "costs":     daily_costs,
        "net_ret":   net_ret,
        "equity":    equity,
        "vix":       data["VIX"] if "VIX" in data.columns else np.nan,
    }).dropna(subset=["net_ret"])

    total_cost_bps = daily_costs.sum() * 10_000
    logger.info(
        f"Backtest complete | {len(result)} days | "
        f"final equity={result['equity'].iloc[-1]:.4f} | "
        f"total costs={total_cost_bps:.1f} bps"
    )
    return result


# ============================================================
# SECTION 6 — Metric helpers + metrics()
# ============================================================

def _sharpe(r: pd.Series, ann: int = 252) -> float:
    """Annualised Sharpe ratio (0% risk-free rate assumption)."""
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0

def _sortino(r: pd.Series, ann: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    down = r[r < 0].std()
    return float(r.mean() / down * np.sqrt(ann)) if down > 0 else 0.0

def _max_drawdown(eq: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    dd = (eq - eq.cummax()) / eq.cummax()
    return float(dd.min())

def _calmar(r: pd.Series, eq: pd.Series, ann: int = 252) -> float:
    """Calmar ratio: annualised return / |max drawdown|."""
    mdd = abs(_max_drawdown(eq))
    return float(r.mean() * ann / mdd) if mdd > 0 else 0.0

def _drawdown_series(eq: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values, same index as eq)."""
    return (eq - eq.cummax()) / eq.cummax()


def metrics(bt: pd.DataFrame, spec: Optional[Dict] = None) -> Dict:
    """
    Full performance metric suite.

    Sections:
      1. Full period     (all available history)
      2. In-sample       (first 70% of days)
      3. Out-of-sample   (last  30% of days)
      4. Regime          (Low VIX <20 / High VIX ≥20)
      5. Meta            (capacity, day counts, turnover)

    Returns dict of scalars — safe to JSON-serialise.
    """
    r  = bt["net_ret"].dropna()
    eq = bt["equity"].dropna()
    n  = len(r)
    sp = int(n * 0.70)

    r_in,  r_out  = r.iloc[:sp],  r.iloc[sp:]
    eq_in, eq_out = eq.iloc[:sp], eq.iloc[sp:]

    # Regime masks
    vix = (
        bt["vix"].reindex(r.index).ffill().fillna(20.0)
        if "vix" in bt.columns
        else pd.Series(20.0, index=r.index)
    )
    lo_mask = vix < 20
    hi_mask = vix >= 20
    r_lo    = r[lo_mask]
    r_hi    = r[hi_mask]
    eq_lo   = (1 + r_lo).cumprod()
    eq_hi   = (1 + r_hi).cumprod()

    ann_vol  = r.std() * np.sqrt(252)
    pos_mean = bt["position"].abs().mean()
    turnover = (
        bt["costs"].sum() / pos_mean if pos_mean > 0 else 0.0
    )

    return {
        # Full period
        "ann_return":     round(r.mean()  * 252 * 100, 2),
        "ann_vol":        round(ann_vol   * 100,        2),
        "sharpe":         round(_sharpe(r),             3),
        "sortino":        round(_sortino(r),            3),
        "calmar":         round(_calmar(r, eq),         3),
        "max_drawdown":   round(_max_drawdown(eq)*100,  2),
        "win_rate":       round((r > 0).mean() * 100,   2),
        "turnover_est":   round(float(turnover),        2),

        # In-sample (70%)
        "in_sharpe":       round(_sharpe(r_in),             3),
        "in_sortino":      round(_sortino(r_in),            3),
        "in_calmar":       round(_calmar(r_in, eq_in),      3),
        "in_max_drawdown": round(_max_drawdown(eq_in)*100,  2),
        "in_ann_return":   round(r_in.mean() * 252 * 100,   2),

        # Out-of-sample (30%)
        "out_sharpe":       round(_sharpe(r_out),             3),
        "out_sortino":      round(_sortino(r_out),            3),
        "out_calmar":       round(_calmar(r_out, eq_out),     3),
        "out_max_drawdown": round(_max_drawdown(eq_out)*100,  2),
        "out_ann_return":   round(r_out.mean() * 252 * 100,   2),

        # Regimes
        "low_vix_sharpe":  round(_sharpe(r_lo), 3) if len(r_lo) > 30 else 0.0,
        "high_vix_sharpe": round(_sharpe(r_hi), 3) if len(r_hi) > 30 else 0.0,
        "low_vix_days":    int(lo_mask.sum()),
        "high_vix_days":   int(hi_mask.sum()),

        # Meta
        "capacity_est_M": round(float(eq.iloc[-1]) * 100, 1),
        "total_days":      n,
        "in_sample_days":  sp,
        "out_sample_days": n - sp,
    }


# ============================================================
# SECTION 7 — generate_report : bundle all artefacts
# ============================================================

_SIGNAL_DESCRIPTIONS = {
    "momentum": (
        "**Time-Series Momentum** goes long when price is above its rolling "
        "SMA, capturing trending / risk-on environments. Best edge in low-VIX "
        "bull markets; suffers whipsaw in choppy, mean-reverting regimes. "
        "Widely used in trend-following CTAs and systematic equity funds."
    ),
    "mean_reversion": (
        "**Z-Score Mean Reversion** buys oversold dips (negative z-score) and "
        "fades overbought rallies (positive z-score) relative to a rolling "
        "mean / std window. Best in range-bound, moderate-VIX environments. "
        "Can suffer large drawdowns during persistent trending or crisis regimes."
    ),
    "vol_targeting": (
        "**Volatility Targeting** scales the portfolio position inversely to "
        "realised volatility to maintain a constant annualised risk budget. "
        "Naturally reduces exposure in high-VIX regimes and increases it "
        "during calm periods. Widely used in risk-parity and CTA overlays."
    ),
}


def generate_report(
    metrics_dict: Dict,
    spec:         Dict,
    bt_result:    pd.DataFrame,
) -> Dict:
    """
    Bundle all pipeline artefacts into a single report dict.

    Keys: spec, metrics, equity_curve, drawdown_series, net_returns,
          position_series, vix_series, strategy_description, generated_at
    """
    eq = bt_result["equity"]
    return {
        "spec":                 spec,
        "metrics":              metrics_dict,
        "equity_curve":         eq,
        "drawdown_series":      _drawdown_series(eq),
        "net_returns":          bt_result["net_ret"],
        "position_series":      bt_result["position"],
        "vix_series":           (
                                    bt_result["vix"]
                                    if "vix" in bt_result.columns
                                    else None
                                ),
        "strategy_description": _SIGNAL_DESCRIPTIONS.get(
                                    spec.get("signal", "momentum"), ""
                                ),
        "generated_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# SECTION 8 — generate_pdf_report : ReportLab institutional PDF
# ============================================================

def generate_pdf_report(report: Dict) -> bytes:
    """
    Institutional-style PDF research note via ReportLab Platypus.
    Returns raw PDF bytes → pass directly to st.download_button().

    Install: pip install reportlab
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units     import inch
    from reportlab.lib           import colors
    from reportlab.platypus      import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.9*inch, rightMargin=0.9*inch,
        topMargin=0.9*inch,  bottomMargin=0.9*inch,
    )
    styles = getSampleStyleSheet()
    story  = []

    # Style definitions
    title_style = ParagraphStyle(
        "QA_Title", parent=styles["Heading1"],
        fontSize=18, spaceAfter=4,
        textColor=colors.HexColor("#0f172a"),
    )
    sub_style = ParagraphStyle(
        "QA_Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#64748b"), spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "QA_Body", parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "QA_H2", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#1e40af"),
        spaceBefore=12, spaceAfter=6,
    )

    m   = report["metrics"]
    sp  = report["spec"]
    now = report["generated_at"]
    desc = report.get("strategy_description", "").replace("**", "")

    # Title block
    story.append(Paragraph("Quant Agent — Strategy Research Note", title_style))
    story.append(Paragraph(
        f"Generated: {now}  |  Universe: {sp.get('universe')}  |  "
        f"Signal: {sp.get('signal')}  |  Lookback: {sp.get('lookback')}d  |  "
        f"Hold: {sp.get('hold')}d  |  Cost: {sp.get('cost_bps')} bps",
        sub_style,
    ))
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#e2e8f0"), spaceAfter=10,
    ))

    # Strategy overview
    story.append(Paragraph("Strategy Overview", h2_style))
    story.append(Paragraph(desc, body_style))

    # Metrics table
    story.append(Paragraph("Performance Metrics", h2_style))
    tdata = [
        ["Metric",            "In-Sample (70%)",          "Out-of-Sample (30%)",         "Full Period"],
        ["Ann. Return",       f"{m['in_ann_return']}%",   f"{m['out_ann_return']}%",     f"{m['ann_return']}%"],
        ["Sharpe Ratio",      str(m["in_sharpe"]),        str(m["out_sharpe"]),          str(m["sharpe"])],
        ["Sortino Ratio",     str(m["in_sortino"]),       str(m["out_sortino"]),         str(m["sortino"])],
        ["Calmar Ratio",      str(m["in_calmar"]),        str(m["out_calmar"]),          str(m["calmar"])],
        ["Max Drawdown",      f"{m['in_max_drawdown']}%", f"{m['out_max_drawdown']}%",   f"{m['max_drawdown']}%"],
        ["Ann. Volatility",   "—",                        "—",                           f"{m['ann_vol']}%"],
        ["Win Rate",          "—",                        "—",                           f"{m['win_rate']}%"],
    ]
    tbl_style = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1,-1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",           (0, 0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("ALIGN",          (1, 0), (-1,-1), "CENTER"),
        ("VALIGN",         (0, 0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1,-1), 5),
    ])
    tbl = Table(tdata, colWidths=[2.1*inch, 1.5*inch, 1.9*inch, 1.5*inch])
    tbl.setStyle(tbl_style)
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Regime breakdown
    story.append(Paragraph("Regime Breakdown (VIX threshold: 20)", h2_style))
    rdata = [
        ["Regime",           "Trading Days",              "Sharpe Ratio"],
        ["Low VIX  (< 20)", str(m["low_vix_days"]),      str(m["low_vix_sharpe"])],
        ["High VIX (>= 20)",str(m["high_vix_days"]),     str(m["high_vix_sharpe"])],
    ]
    rtbl = Table(rdata, colWidths=[2.5*inch, 2*inch, 2*inch])
    rtbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1,-1),
         [colors.HexColor("#eff6ff"), colors.white]),
        ("GRID",           (0, 0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("ALIGN",          (1, 0), (-1,-1), "CENTER"),
        ("VALIGN",         (0, 0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1,-1), 5),
    ]))
    story.append(rtbl)
    story.append(Spacer(1, 16))

    # Disclaimer
    disc_style = ParagraphStyle(
        "QA_Disc", parent=styles["Normal"],
        fontSize=7, textColor=colors.HexColor("#94a3b8"),
    )
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#e2e8f0"), spaceBefore=8,
    ))
    story.append(Paragraph(
        "DISCLAIMER: For informational and research purposes only. "
        "Past performance is not indicative of future results. "
        "Not investment advice. Generated by Quant Agent v1.0.",
        disc_style,
    ))

    doc.build(story)
    return buf.getvalue()


# ============================================================
# SECTION 9 — generate_html_report : self-contained HTML report
# ============================================================

def generate_html_report(report: Dict) -> str:
    """
    Build a self-contained HTML research report (inline CSS, no external deps).
    Returns HTML string → encode to bytes for st.download_button().
    """
    m    = report["metrics"]
    sp   = report["spec"]
    now  = report["generated_at"]
    desc = report.get("strategy_description", "").replace("**", "")

    def row(label, in_v, out_v, full_v):
        return (
            f"<tr><td>{label}</td>"
            f"<td class='num'>{in_v}</td>"
            f"<td class='num'>{out_v}</td>"
            f"<td class='num'>{full_v}</td></tr>"
        )

    metric_rows = "".join([
        row("Ann. Return",   f"{m['in_ann_return']}%",  f"{m['out_ann_return']}%",  f"{m['ann_return']}%"),
        row("Sharpe",         m['in_sharpe'],             m['out_sharpe'],             m['sharpe']),
        row("Sortino",        m['in_sortino'],            m['out_sortino'],            m['sortino']),
        row("Calmar",         m['in_calmar'],             m['out_calmar'],             m['calmar']),
        row("Max Drawdown",  f"{m['in_max_drawdown']}%", f"{m['out_max_drawdown']}%", f"{m['max_drawdown']}%"),
        row("Ann. Vol",       "—",                        "—",                         f"{m['ann_vol']}%"),
        row("Win Rate",       "—",                        "—",                         f"{m['win_rate']}%"),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Quant Agent Research Report</title>
  <style>
    *     {{ box-sizing:border-box; margin:0; padding:0; }}
    body  {{ font-family:'Segoe UI',Arial,sans-serif; background:#f8fafc;
             color:#1e293b; padding:40px 60px; max-width:960px; margin:auto; }}
    h1    {{ font-size:26px; color:#0f172a; margin-bottom:4px; }}
    .sub  {{ font-size:12px; color:#64748b; margin-bottom:28px; }}
    h2    {{ font-size:16px; margin:28px 0 10px; color:#0f172a;
             border-left:4px solid #3b82f6; padding-left:10px; }}
    .desc {{ font-size:13px; line-height:1.7; color:#334155;
             background:#eff6ff; padding:14px 18px;
             border-radius:8px; margin-bottom:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px;
             margin-bottom:20px; }}
    thead {{ background:#0f172a; color:white; }}
    thead th {{ padding:10px 14px; text-align:left; }}
    tbody tr:nth-child(even) {{ background:#f1f5f9; }}
    tbody td {{ padding:9px 14px; border-bottom:1px solid #e2e8f0; }}
    .num  {{ text-align:center; font-family:monospace; }}
    .badge{{ display:inline-block; background:#3b82f6; color:white;
             border-radius:4px; padding:2px 8px; font-size:11px;
             margin-right:6px; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr);
                  gap:12px; margin-bottom:24px; }}
    .kpi  {{ background:#1e293b; color:white; border-radius:10px;
             padding:14px 18px; border-left:4px solid #3b82f6; }}
    .kpi-val  {{ font-size:24px; font-weight:700; color:#60a5fa; }}
    .kpi-lbl  {{ font-size:11px; color:#94a3b8; margin-top:4px; }}
    .footer{{ margin-top:40px; font-size:10px; color:#94a3b8;
              border-top:1px solid #e2e8f0; padding-top:12px; }}
  </style>
</head>
<body>
  <h1>&#9889; Quant Agent &mdash; Strategy Research Note</h1>
  <p class="sub">
    Generated: {now} &nbsp;|&nbsp;
    <span class="badge">{sp.get('universe')}</span>
    <span class="badge">{sp.get('signal')}</span>
    Lookback: {sp.get('lookback')}d &nbsp;|&nbsp;
    Hold: {sp.get('hold')}d &nbsp;|&nbsp;
    Cost: {sp.get('cost_bps')} bps
  </p>

  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-val">{m['sharpe']}</div>
        <div class="kpi-lbl">Sharpe Ratio</div></div>
    <div class="kpi"><div class="kpi-val">{m['max_drawdown']}%</div>
        <div class="kpi-lbl">Max Drawdown</div></div>
    <div class="kpi"><div class="kpi-val">{m['ann_return']}%</div>
        <div class="kpi-lbl">Ann. Return</div></div>
    <div class="kpi"><div class="kpi-val">{m['win_rate']}%</div>
        <div class="kpi-lbl">Win Rate</div></div>
  </div>

  <h2>Strategy Overview</h2>
  <div class="desc">{desc}</div>

  <h2>Performance Metrics</h2>
  <table>
    <thead><tr>
      <th>Metric</th><th>In-Sample (70%)</th>
      <th>Out-of-Sample (30%)</th><th>Full Period</th>
    </tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>

  <h2>Regime Breakdown (VIX threshold: 20)</h2>
  <table>
    <thead><tr><th>Regime</th><th>Trading Days</th><th>Sharpe</th></tr></thead>
    <tbody>
      <tr><td>Low VIX (&lt;20)</td>
          <td class="num">{m['low_vix_days']}</td>
          <td class="num">{m['low_vix_sharpe']}</td></tr>
      <tr><td>High VIX (&ge;20)</td>
          <td class="num">{m['high_vix_days']}</td>
          <td class="num">{m['high_vix_sharpe']}</td></tr>
    </tbody>
  </table>

  <p class="footer">
    DISCLAIMER: For informational &amp; research purposes only.
    Past performance is not indicative of future results.
    Not investment advice. &nbsp;|&nbsp; Generated by Quant Agent v1.0.
  </p>
</body>
</html>"""


# ============================================================
# SECTION 10 — run_full_pipeline : one-call entry point
# ============================================================

def run_full_pipeline(
    prompt: str,
    start:  str = DEFAULT_START,
    end:    str = DEFAULT_END,
) -> Dict:
    """
    One-call pipeline: English prompt → full report dict.

    Usage:
        report = run_full_pipeline("60-day momentum on SPY")
        report = run_full_pipeline("mean reversion SP500", start="2018-01-01")

    Returns dict with keys:
        spec, metrics, equity_curve, drawdown_series, net_returns,
        position_series, vix_series, strategy_description, generated_at,
        backtest_df  (raw DataFrame for Streamlit table rendering)
    """
    logger.info(f"Pipeline start | prompt='{prompt}'")

    spec              = parse_english(prompt)
    data, data_sources = fetch_data(spec, start=start, end=end)
    signal            = build_signal(data, spec)
    bt                = backtest(data, signal, spec=spec)
    m_dict            = metrics(bt, spec)
    report            = generate_report(m_dict, spec, bt)
    report["backtest_df"]   = bt            # raw df for app.py table
    report["data_sources"]  = data_sources  # FMP/FRED status for UI warnings

    logger.info(
        f"Pipeline complete | "
        f"Sharpe={m_dict['sharpe']} | "
        f"MaxDD={m_dict['max_drawdown']}% | "
        f"AnnReturn={m_dict['ann_return']}%"
    )
    return report


# ============================================================
# Quick smoke-test (python quant_engine.py)
# ============================================================
if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)

    print("\n=== parse_english tests ===")
    for t in [
        "60-day momentum on SPY",
        "mean reversion SP500",
        "vol targeting on QQQ",
        "20-day momentum on SP500 large caps with 10bps costs",
        "short-term mean reversion on QQQ with 5bps costs",
    ]:
        print(f"  '{t}'\n    → {parse_english(t)}\n")

    print("=== Full pipeline test (SPY momentum 2020-2024) ===")
    report = run_full_pipeline("60-day momentum on SPY", start="2020-01-01", end="2024-12-31")
    m = report["metrics"]
    print(f"  Sharpe   : {m['sharpe']}")
    print(f"  MaxDD    : {m['max_drawdown']}%")
    print(f"  Ann Ret  : {m['ann_return']}%")
    print(f"  Win Rate : {m['win_rate']}%")

    html = generate_html_report(report)
    with open("reports/test_report.html", "w") as f:
        f.write(html)
    print(f"  HTML → reports/test_report.html ({len(html):,} chars)")

    try:
        pdf = generate_pdf_report(report)
        with open("reports/test_report.pdf", "wb") as f:
            f.write(pdf)
        print(f"  PDF  → reports/test_report.pdf ({len(pdf):,} bytes)")
    except ImportError:
        print("  PDF skipped — pip install reportlab")

    print("\n✅ quant_engine.py smoke-test complete.")