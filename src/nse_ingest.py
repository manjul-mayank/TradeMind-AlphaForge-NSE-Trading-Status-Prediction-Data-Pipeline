"""
Data ingestion for NSE India (unofficial). 
Notes:
- These endpoints are for educational/demo use and may change or be rate-limited by NSE.
- For robust production ingestion, consider licensed feeds or a broker API (Zerodha/Angel/Upstox).
"""
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import StringIO
from tqdm import tqdm
from .utils import ensure_dir, save_csv, add_symbol, chronological

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

def _session():
    s = requests.Session()
    s.headers.update(HEADERS)
    # Warm-up request to set cookies
    s.get("https://www.nseindia.com/")
    return s

def fetch_equity_bhavcopy_for_date(date: datetime) -> pd.DataFrame:
    """
    Download daily equity bhavcopy CSV for a given date.
    Example URL: https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
    """
    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date.strftime('%d%m%Y')}.csv"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [col.strip() for col in df.columns]  # Remove leading/trailing spaces
    print(f"[DEBUG] Columns in bhavcopy for {date.strftime('%Y-%m-%d')}: {df.columns.tolist()}")
    # Normalize columns to match actual CSV
    df.rename(columns={
        "SYMBOL": "symbol",
        "SERIES": "series",
        "DATE1": "date",
        "PREV_CLOSE": "prev_close",
        "OPEN_PRICE": "open",
        "HIGH_PRICE": "high",
        "LOW_PRICE": "low",
        "LAST_PRICE": "last",
        "CLOSE_PRICE": "close",
        "TTL_TRD_QNTY": "volume",
        "TURNOVER_LACS": "turnover_lacs",
        "NO_OF_TRADES": "num_trades",
        "DELIV_QTY": "deliv_qty",
        "DELIV_PER": "deliv_per"
    }, inplace=True)
    print(f"[DEBUG] Columns after rename: {df.columns.tolist()}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        print(f"[ERROR] 'date' column not found after renaming for {date.strftime('%Y-%m-%d')}")
    return df

def fetch_recent_index_snapshot(index_name="NIFTY 50") -> pd.DataFrame:
    """
    Snapshot of constituents with LTP/Day High/Low etc.
    """
    s = _session()
    params = {"index": index_name}
    url = "https://www.nseindia.com/api/equity-stockIndices"
    r = s.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Keep relevant fields
    keep = ["symbol","open","dayHigh","dayLow","lastPrice","pChange","totalTradedVolume"]
    df = df[keep]
    df.rename(columns={"dayHigh":"high","dayLow":"low","lastPrice":"close","totalTradedVolume":"volume"}, inplace=True)
    df["date"] = pd.to_datetime(datetime.now().date())
    return df

def ingest_symbols_from_bhavcopy(start_date: str, end_date: str, symbols: list, out_dir: str) -> None:
    """
    Loop through date range, download bhavcopy, filter for symbols, append to per-symbol CSVs.
    """
    ensure_dir(out_dir)
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    for d in tqdm(pd.date_range(start, end, freq="D")):
        df = fetch_equity_bhavcopy_for_date(d.to_pydatetime())
        if df.empty: 
            continue
        for sym in symbols:
            sub = df[df["symbol"]==sym]
            if sub.empty: 
                continue
            sub = add_symbol(sub, sym)
            path = os.path.join(out_dir, f"{sym}.csv")
            # Append or create
            if os.path.exists(path):
                existing = pd.read_csv(path)
                merged = pd.concat([existing, sub], ignore_index=True)
                merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
                merged = merged.drop_duplicates(subset=["date","symbol"]).sort_values("date")
                merged.to_csv(path, index=False)
            else:
                sub.to_csv(path, index=False)
        time.sleep(0.2)  # be polite
