import pandas as pd
import numpy as np


class BaseLongStrategy:
    """
    Template class for Long-only trading strategies.

    Methods
    -------
    generate_features:
        generates the features to be used for the strategy.
    define_position:
        calculates the positions for the strategy using the features.
    """
    def generate_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        features_df = ohlcv_df.copy()
        features_df["returns"] = np.log(features_df['Close'] / features_df['Close'].shift())
        return features_df

    def define_position(self, features_df: pd.DataFrame, params: dict) -> pd.DataFrame:
        features_df['position'] = 1  # Simplest buy-and-hold strategy
        return features_df
