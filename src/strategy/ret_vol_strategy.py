from src.strategy.base_long_strategy import BaseLongStrategy
import pandas as pd
import numpy as np


class RetVolStrategy(BaseLongStrategy):
    """
    Return volume strategy.
    """
    def generate_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        features_df = ohlcv_df.copy()
        # Ignore divide by zero errors as they will be handled below
        with np.errstate(divide='ignore'):
            features_df["returns"] = np.log(features_df['Close'] / features_df['Close'].shift())
            features_df["vol_ch"] = np.log(features_df['Volume'] / features_df['Volume'].shift())

        # Outliers removal
        features_df.loc[features_df["vol_ch"] > 3, "vol_ch"] = np.nan
        features_df.loc[features_df["vol_ch"] < -3, "vol_ch"] = np.nan
        return features_df

    def define_position(self, features_df: pd.DataFrame, params: dict) -> pd.DataFrame:
        features_df['position'] = 1  # Simplest buy-and-hold strategy
        # Condition 1
        return_thresh = np.percentile(features_df["returns"].dropna(), params["p_ret"])
        cond1 = features_df["returns"] >= return_thresh
        # Condition 2
        volume_thresh = np.percentile(features_df["vol_ch"].dropna(), [params["p_vol_low"], params["p_vol_high"]])
        cond2 = features_df["vol_ch"].between(volume_thresh[0], volume_thresh[1])

        # Joint conditions
        features_df.loc[cond1 & cond2, "position"] = 0
        return features_df
