from src.strategy.base_long_strategy import BaseLongStrategy
from talib import RSI
import pandas as pd
import numpy as np


class RSIStrategy(BaseLongStrategy):
    """
    RSI-based strategy.
    """
    def generate_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        features_df = ohlcv_df.copy()
        # Ignore divide by zero errors as they will be handled below
        with np.errstate(divide='ignore'):
            features_df["returns"] = np.log(features_df['Close'] / features_df['Close'].shift())
            features_df["vol_ch"] = np.log(features_df['Volume'] / features_df['Volume'].shift())
            features_df["rsi"] = RSI(features_df['Close'], timeperiod=14)

        # Outliers removal
        features_df.loc[features_df["vol_ch"] > 3, "vol_ch"] = np.nan
        features_df.loc[features_df["vol_ch"] < -3, "vol_ch"] = np.nan
        return features_df

    def define_position(self, features_df: pd.DataFrame, params: dict) -> pd.DataFrame:
        features_df['position'] = 1  # Simplest buy-and-hold strategy

        # Conditions
        cond1 = features_df["rsi"] >= params['rsi_thresh_low']
        cond2 = features_df["rsi"] <= params['rsi_thresh_high']
        cond3 = features_df["returns"] >= params['ret_thresh']
        features_df.loc[cond1 & cond2 & cond3, "position"] = 0
        return features_df
