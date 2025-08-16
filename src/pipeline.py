
"""
High-level orchestration pipeline:
1) Ingest historical data from NSE archives for configured symbols
2) Build features & labels
3) Train model and save artifact
4) Generate simple report
"""
import os, yaml
import pandas as pd
from datetime import datetime, timedelta
from .nse_ingest import ingest_symbols_from_bhavcopy
from .feature_engineering import build_features
from .labeling import make_labels
from .model_train import train_model, save_model
from .backtest import backtest_signals
from .utils import ensure_dir

def run(config_path="./config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["symbols"]
    lookback_days = int(cfg.get("lookback_days", 120))
    today = datetime.today().date()
    start_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    raw_dir = cfg["storage"]["raw_dir"]
    processed_dir = cfg["storage"]["processed_dir"]
    models_dir = cfg["storage"]["models_dir"]
    reports_dir = cfg["storage"]["reports_dir"]
    ensure_dir(raw_dir); ensure_dir(processed_dir); ensure_dir(models_dir); ensure_dir(reports_dir)

    # 1) Ingest
    print(f"Ingesting bhavcopy for {symbols} from {start_date} to {end_date} ...")
    ingest_symbols_from_bhavcopy(start_date, end_date, symbols, raw_dir)

    # 2) Build per-symbol dataset
    frames = []
    for sym in symbols:
        path = os.path.join(raw_dir, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"Warning: No data for {sym}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df.assign(symbol=sym))
    if not frames:
        raise RuntimeError("No data found; ingestion likely failed.")
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["symbol","date"]).reset_index(drop=True)

    # 3) Features & labels (per symbol, then concat)
    feat_frames = []
    for sym, sub in data.groupby("symbol"):
        f = build_features(sub)
        feat_frames.append(f)
    feats = pd.concat(feat_frames, ignore_index=True)

    labeled = make_labels(feats, horizon_days=cfg["labeling"]["horizon_days"],
                          task=cfg["labeling"]["task"],
                          threshold_pct=cfg["labeling"]["threshold_pct"])

    # 4) Train model
    model, cv_scores, feature_names = train_model(labeled,
                                                  task=cfg["labeling"]["task"],
                                                  model_type=cfg["model"]["type"],
                                                  params=cfg["model"]["params"])
    print("CV scores:", cv_scores)
    model_path = os.path.join(models_dir, f"model_{cfg['model']['type']}_{cfg['labeling']['task']}.joblib")
    save_model(model, feature_names, model_path)
    print(f"Saved model to {model_path}")

    # 5) Generate signals (in-sample simple demonstration)
    if cfg["labeling"]["task"] == "classification":
        X = labeled[feature_names]
        labeled["signal"] = model.predict(X)
    else:
        X = labeled[feature_names]
        yhat = model.predict(X)
        # convert regression predictions into signals with same thresholds
        up = (yhat >= cfg["labeling"]["threshold_pct"]).astype(int)
        dn = (yhat <= -cfg["labeling"]["threshold_pct"]).astype(int) * -1
        labeled["signal"] = up + dn

    # 6) Backtest (naive, no train/test separation here; for illustration)
    bt = backtest_signals(labeled, signal_col="signal", price_col="close")
    report_path = os.path.join(reports_dir, "equity_curve.csv")
    bt.to_csv(report_path, index=False)
    print(f"Backtest equity curve saved to {report_path}")

if __name__ == "__main__":
    run()
