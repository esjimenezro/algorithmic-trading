from src.backtesting.long_only_backtesting import LongOnlyBacktester
from src.strategy.ret_vol_strategy import RetVolStrategy
from src.strategy.rsi_strategy import RSIStrategy
import numpy as np


if __name__ == "__main__":

    bt_ret_vol = LongOnlyBacktester(
        strategy=RetVolStrategy(),
        symbol="BTCUSDT",
        time_frame="1h",
        start="2020-03-01",
        # start="2017-08-01",
        end="2022-03-01",
        #  end="2020-03-31",
        tc=-0.0011
    )
    """
    bt_ret_vol.optimize_strategy(param_ranges={
        "p_ret": range(85, 98),
        "p_vol_low": range(2, 16),
        "p_vol_high": range(16, 35)}
    )
    """

    bt_ret_vol.test_strategy(parameters={'p_ret': 89, 'p_vol_low': 12, 'p_vol_high': 28})

    bt_ret_vol.plot_results()
