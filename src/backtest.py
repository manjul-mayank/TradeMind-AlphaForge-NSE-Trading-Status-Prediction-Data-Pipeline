
import pandas as pd
import numpy as np

def backtest_signals(df: pd.DataFrame, signal_col="signal", price_col="close", fee_bps=5):
    """
    Simple daily close-to-close backtest with flat fee per trade (in bps).
    signal: -1 sell, 0 hold, 1 buy, applied at close for next day's return.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["ret_1d"] = df[price_col].pct_change()
    df["shift_sig"] = df[signal_col].shift(1).fillna(0)

    # Strategy return
    strat_ret = df["shift_sig"] * df["ret_1d"]
    # Fees when signal changes
    trade = (df["shift_sig"].diff().abs() > 0).astype(int)
    fees = trade * (fee_bps / 10000.0)
    strat_ret_after_fee = strat_ret - fees
    df["strat_ret"] = strat_ret_after_fee.fillna(0.0)
    df["equity_curve"] = (1.0 + df["strat_ret"]).cumprod()
    return df
