# Quant Agent

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)


**AI-powered systematic strategy research platform** — type English prompts like `"60-day momentum on SPY"`, get institutional-grade backtests, regime analysis, and research reports in seconds.

## 🎬 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-agent.streamlit.app)

## 🚀 Features

| Feature | Status |
|---|---|
| **Natural Language Parser** | ✅ `"20-day momentum SP500 with 10bps costs"` |
| **3 Signal Types** | ✅ Momentum · Mean Reversion · Vol Targeting |
| **Multi-Asset Universes** | ✅ SPY · SP500 · QQQ · IWM · DIA |
| **Walk-Forward Validation** | ✅ 70/30 IS/OOS split (zero lookahead) |
| **Risk Metrics** | ✅ Sharpe · Sortino · Calmar · Max DD |
| **VIX Regime Analysis** | ✅ Low/High VIX Sharpe breakdown |
| **Equity Curves** | ✅ Plotly dual-pane + IS/OOS split |
| **Virtual P&L Table** | ✅ Last 30 days with costs |
| **Institutional Exports** | ✅ PDF · HTML · JSON metrics |

## 📊 Sample Output

```
Prompt: "60-day momentum on SPY with 5bps costs"
Sharpe: 0.64 | Max DD: -9.5% | Ann. Return: 0.26%
IS Sharpe: 0.615 → OOS Sharpe: 0.704 (degradation: -0.089 ✅)
Low VIX Sharpe: 0.737 | High VIX Sharpe: 0.073
```

## 🏗 Architecture

```
English Prompt
     ↓ parse_english()
Strategy Spec Dict
     ↓ fetch_data() [FMP + FRED]
OHLCV + VIX Data
     ↓ build_signal()
Signal Series
     ↓ backtest()
P&L + Equity Curve
     ↓ metrics()
Full Risk Metrics
     ↓ Streamlit Dashboard + PDF Export
```

## ⚙️ Quick Start

```bash
git clone https://github.com/yourusername/quant-agent
cd quant-agent

# Install dependencies
pip install -r requirements.txt

# Get free FMP API key (250 req/day free tier)
# https://site.financialmodelingprep.com/developer
echo "FMP_API_KEY=your_key_here" > .env

# Run locally
streamlit run app.py
```

## 📦 Installation

```bash
pip install streamlit pandas numpy plotly requests reportlab python-dotenv
```

**Optional API Keys** (stored in `.env`):

```bash
# Free tier FMP (250 req/day) - required for real data
FMP_API_KEY=your_fmp_key_here

# FRED (unlimited) - VIX data  
FRED_API_KEY=optional_fred_key
```

No API key? Engine auto-falls back to synthetic GBM data.

## 🎯 Test Strategies

Copy-paste these into the prompt box:

```text
# Beginner
60-day momentum on SPY

# Intermediate  
20-day mean reversion on QQQ with 5bps costs

# Advanced
vol targeting on SPY with 10bps costs

# Stress test
40-day momentum on SP500 large caps with 10bps costs
```

## 🛠 File Structure

```
quant-agent/
├── app.py                 # Streamlit dashboard (production-ready)
├── quant_engine.py        # Core engine: parse → backtest → metrics → export
├── requirements.txt       # Dependencies
├── .env.example          # API keys template
├── README.md             # This file
└── reports/              # Generated PDFs (gitignored)
```

## 🔧 Customization

**Extend signals** (`quant_engine.py`):
```python
_SIGNAL_DESCRIPTIONS["carry"] = "Carry trades long high-yield..."
def build_signal(data, spec):
    if spec["signal"] == "carry":
        return data["close"].pct_change(252).rolling(252).mean()
```

**Add universes**:
```python
if "btc" in prompt: spec["universe"] = "BTCUSD"
```

## 📈 Expected Metrics

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| **Sharpe** | > 1.0 | 0.5–1.0 | < 0.5 |
| **Max DD** | < 15% | 15–25% | > 25% |
| **OOS Degradation** | < 0.3 | 0.3–0.6 | > 0.6 |
| **Low/High VIX Sharpe** | Both > 0.3 | | One negative |

## 🌐 Deployment

**Streamlit Cloud** (free):
1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add `FMP_API_KEY` to `.streamlit/secrets.toml`

**Docker**:
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## 📚 Data Sources

| Provider | Data | Rate Limit | Cost |
|---|---|---|---|
| [FMP Stable](https://financialmodelingprep.com) | OHLCV (SPY, SP500, etc.) | 250 req/day (free) | Free |
| [FRED](https://fred.stlouisfed.org) | VIX daily | Unlimited | Free |

## 🤝 Contributing

1. Fork repo
2. Add new signals to `build_signal()` in `quant_engine.py`
3. Add to `EXAMPLE_MAPPINGS` for natural language parsing
4. Test with `streamlit run app.py`
5. PR with example PDF output

## 📄 License

MIT — free for commercial use.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io) — reactive UIs
- [Plotly](https://plotly.com/python) — publication charts
- [FMP](https://financialmodelingprep.com) — market data
- [ReportLab](https://www.reportlab.com) — PDF export

---

*Past performance does not guarantee future results. Not financial advice. For research purposes only.*

---
```