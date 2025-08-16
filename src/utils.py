
import os
import pandas as pd
from datetime import datetime

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_date(date_str: str) -> pd.Timestamp:
    return pd.to_datetime(date_str).tz_localize(None)

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)

def save_csv(df: pd.DataFrame, path: str, index=False) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)

def add_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    else:
        df["symbol"] = df["symbol"].fillna(symbol)
    return df

def chronological(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    return df.sort_values(date_col).reset_index(drop=True)
