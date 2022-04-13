import pandas as pd


class BaseLongStrategy:

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def define_position(self, df: pd.DataFrame) -> pd.DataFrame:
        df['position'] = 1  # Simplest buy-and-hold strategy
        return df