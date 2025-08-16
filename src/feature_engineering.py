
import os
import pandas as pd
from .indicators import sma, ema, rsi, macd, bollinger_bands, atr

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    # Basic price features
    df["ret_1d"] = df["close"].pct_change()*100
    df["ret_5d"] = df["close"].pct_change(5)*100
    df["vwap_proxy"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # Indicators
    df["sma_5"] = sma(df["close"], 5)
    df["sma_20"] = sma(df["close"], 20)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["rsi_14"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"], 12, 26, 9)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    ma, upper, lower = bollinger_bands(df["close"], 20, 2)
    df["bb_mid"] = ma
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    # Lags
    for lag in [1,2,3,5,10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)

    # Drop earliest rows with NaNs due to rolling calculations
    df = df.dropna().reset_index(drop=True)
    return df
