# TradeMind AlphaForge: NSE Trading Status & Prediction Data Pipeline  

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![Build](https://img.shields.io/badge/build-passing-brightgreen.svg) ![License](https://img.shields.io/badge/license-MIT-yellow.svg) ![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)  

A modular **AI-powered quantitative trading pipeline** built for **NSE equity data**. This project ingests historical stock data, engineers technical indicators, labels signals, trains ML models, backtests performance, and generates equity curve reports.  

⚡ With **TradeMind AlphaForge**, you can go from **raw market data → predictive trading signals → backtested equity curves** in a single automated pipeline.   

![Pipeline](./Pipeline%20image.png)

---

## 🚀 Features  

- **Data Ingestion**  
  - Fetches **NSE equity bhavcopy** (unofficial endpoints).  
  - Saves raw CSVs per symbol in `data/raw/`.  

- **Feature Engineering**  
  - Calculates powerful **technical indicators**:  
    - SMA, EMA, RSI, MACD, Bollinger Bands, ATR  
    - Price lags and statistical features  
  - Prepares `data/processed/` datasets for ML.  

- **Labeling**  
  - Creates **Buy / Hold / Sell** classification labels.  
  - Supports regression targets for **next-day returns**.  

- **Model Training**  
  - Baseline models: **Random Forest**, **Logistic Regression**, **SVM**.  
  - Saves trained models in `models/`.  

- **Backtesting**  
  - Runs equity curve simulation using predicted signals.  
  - Outputs **PnL reports** in `reports/equity_curve.csv`.  

- **Reports & Explainability**  
  - Generates **equity curve visualizations**.  
  - Future-ready for **SHAP/LIME feature explainability**.  

---

> ⚠️ The NSE endpoints in this demo are **unofficial** and may change or rate-limit traffic.
> For production-grade, use **licensed data feeds** or **broker APIs** (Zerodha, Upstox, Angel One).

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline
```

### Configure
Edit `config/config.yaml` to pick symbols, features, label type, and model.

### Output
- Raw CSVs per symbol in `data/raw/`
- Processed/engineered data in `data/processed/` (optional to add)
- Trained model in `models/`
- Backtest equity curve in `reports/equity_curve.csv`

---

## Project Structure

```
trading-pipeline/
  ├─ config/
  │   └─ config.yaml
  ├─ data/
  │   ├─ raw/
  │   └─ processed/
  ├─ models/
  ├─ reports/
  ├─ src/
  │   ├─ indicators.py
  │   ├─ feature_engineering.py
  │   ├─ labeling.py
  │   ├─ model_train.py
  │   ├─ backtest.py
  │   ├─ nse_ingest.py
  │   └─ utils.py
  ├─ requirements.txt
  └─ README.md
```

---
## 📊 Pipeline Flow

- Step-by-step flow:

 - Data Ingestion → NSE Bhavcopy

 - Feature Engineering → SMA, EMA, RSI, MACD, ATR, Bollinger

 - Labeling → Buy / Hold / Sell, Next-day return

 - Model Training → ML classifiers/regressors

 - Backtesting → Equity curve, PnL simulation

 - Reports & Outputs → CSVs, models, performance plots
---
## 📈 Sample Equity Curve

The pipeline generates equity_curve.csv, which can be plotted for strategy performance.

- Example output:

 - Sharpe ratio, cumulative return, drawdown analysis

 - Equity curve visualization for backtest period

## 🔮 Future Enhancements

- Scheduling → Automate daily runs via cron/Task Scheduler.

- Persistence → Store results in SQLite/Postgres.

- Validation → Add walk-forward testing.

- Live Signals → Plug into broker APIs (Zerodha, Upstox, Angel One).

- Risk Management → Stop-loss, take-profit, position sizing.

- Explainability → SHAP values, feature importance logs.

- LLM/RAG Integration → Add news sentiment signals (headlines → embeddings → event-driven trading).

---
## ⚠️ Disclaimer

**This project is for educational & research purposes only.**
- It is not financial advice and should not be used to make live trading decisions without thorough testing, compliance, and licensed data feeds.
