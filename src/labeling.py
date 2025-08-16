
import pandas as pd

def make_labels(df: pd.DataFrame, horizon_days: int = 1, task: str = "classification", threshold_pct: float = 0.5):
    """
    Adds target columns:
    - For regression: y_reg = future return (%) over 'horizon_days'
    - For classification: y_cls in { -1 (sell), 0 (hold), +1 (buy) } based on threshold_pct
    """
    df = df.copy()
    future = df["close"].shift(-horizon_days)
    y_ret = (future/df["close"] - 1.0)*100.0
    df["y_ret"] = y_ret

    if task == "regression":
        df["y_reg"] = y_ret
    else:
        # classification
        buy = (y_ret >= threshold_pct).astype(int)
        sell = (y_ret <= -threshold_pct).astype(int)*-1
        hold = (~((y_ret >= threshold_pct) | (y_ret <= -threshold_pct))).astype(int)*0
        # combine; priority buy/sell over hold
        y = buy + sell + hold
        df["y_cls"] = y

    # Remove the last 'horizon_days' rows that don't have future info
    return df.iloc[:-horizon_days].reset_index(drop=True)
