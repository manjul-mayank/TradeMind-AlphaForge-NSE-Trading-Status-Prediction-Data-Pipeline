# Trading Status & Prediction Data Pipeline (NSE)

A ready-to-run, modular pipeline that:
- Ingests **NSE equity bhavcopy** data (unofficial endpoints).
- Engineers robust **technical features** (SMA/EMA/RSI/MACD/Bollinger/ATR, lags).
- Creates **labels** for classification (buy/hold/sell) or regression (next-day return).
- Trains a baseline **ML model** (RandomForest/LogReg).
- Runs a simple **backtest** and saves an equity curve report.

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

## Notes & Next Steps

1. **Scheduling**: Use cron or Windows Task Scheduler to run daily after market close.
   - Example cron: `15 17 * * 1-5 /usr/bin/python -m src.pipeline`
2. **Persistence**: Add SQLite/Postgres write steps for data & predictions.
3. **Better Validation**: Add **walk-forward** train/test split and out-of-sample backtest.
4. **Live Signals**: Replace archives with broker/websocket feeds for intraday signals.
5. **Risk**: Add stop-loss/take-profit, position sizing, and transaction cost modeling.
6. **Explainability**: Log feature importances, SHAP values for decisions.
7. **LLM/RAG**: Enrich signals with **news sentiment** (headlines → embeddings) for event-driven trading.

---

## Disclaimer
This code is for **educational purposes** only. It is **not financial advice** and should not be used to place trades without further testing and regulatory compliance.
