
import os
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from .feature_engineering import build_features
from .labeling import make_labels

def train_model(df: pd.DataFrame, task="classification", model_type="random_forest", params=None):
    df_feat = build_features(df)
    labeled = make_labels(df_feat, task=task, horizon_days=1, threshold_pct=0.5)

    # Feature set: drop non-features
    drop_cols = ["date","symbol","series","last","prev_close","turnover","y_ret"]
    y_col = "y_cls" if task=="classification" else "y_reg"
    X = labeled.drop(columns=[c for c in drop_cols if c in labeled.columns] + [y_col], errors="ignore")
    y = labeled[y_col]

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    best_model = None
    best_score = -1e9

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if task=="classification":
            if model_type=="logreg":
                model = LogisticRegression(max_iter=500, n_jobs=None)
            else:
                model = RandomForestClassifier(**(params or {"n_estimators":300,"max_depth":6,"random_state":42}))
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average="macro")
        else:
            if model_type=="logreg":
                # Not typical for regression; fallback to RF Regressor
                model = RandomForestRegressor(**(params or {"n_estimators":300,"max_depth":6,"random_state":42}))
            else:
                model = RandomForestRegressor(**(params or {"n_estimators":300,"max_depth":6,"random_state":42}))
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = -mean_absolute_error(y_val, preds)

        scores.append(score)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, scores, X.columns.tolist()

def save_model(model, feature_names, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": model, "features": feature_names}, out_path)
