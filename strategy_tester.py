from src.backtesting.long_only_backtesting import LongOnlyBacktester
from src.strategy.ret_vol_strategy import RetVolStrategy


if __name__ == "__main__":

    bt_ret_vol = LongOnlyBacktester(
        strategy=RetVolStrategy(),
        symbol="BTCUSDT",
        time_frame="1h",
        start="2017-08-01",
        end="2022-02-28",
        tc=-0.0011
    )

    bt_ret_vol.optimize_strategy(param_ranges={
        "p_ret": range(85, 98),
        "p_vol_low": range(2, 16),
        "p_vol_high": range(16, 35)}
    )

    bt_ret_vol.plot_results()
