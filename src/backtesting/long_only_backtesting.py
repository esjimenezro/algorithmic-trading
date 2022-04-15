from src.strategy.base_long_strategy import BaseLongStrategy
from matplotlib import pyplot as plt
from urllib.error import HTTPError
from itertools import product
import pandas as pd
import numpy as np


class LongOnlyBacktester:
    """
    Class for the vectorized backtesting of Long-only trading
    strategies.

    Attributes
    ----------
    strategy : BaseLongStrategy
        strategy to evaluate.
    symbol : str
        symbol to apply the strategy to.
    time_frame : str
        one of the valid time frames.
    start : str
        start date for data import.
    end : str
        end date for data import.
    tc : float
        proportional trading costs per trade.
    ohlcv_data : pd.DataFrame
        OHLCV data read from data_filepath.
    position_data: pd.DataFrame
        data with features and positions.
    opt_results: pd.DataFrame
        data with optimization results.
    best_params: dict
        data with features and positions.

    Methods
    -------
    get_data:
        imports the data.
    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).
    prepare_data:
        prepares the data for backtesting.
    run_backtest:
        runs the strategy backtest.
    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.
    optimize_strategy:
        backtests strategy for different parameter values incl. optimization and reporting (wrapper).
    find_best_strategy:
        finds the optimal strategy (global maximum).
    print_performance:
        calculates and prints various performance metrics.
    """

    def __init__(self,
                 strategy: BaseLongStrategy,
                 symbol: str,
                 time_frame: str,
                 start: str,
                 end: str,
                 tc: float) -> None:
        """
        Constructor method for the LongOnlyBacktester class.

        Parameters
        ----------
        strategy : BaseLongStrategy
            strategy to evaluate.
        symbol : str
            symbol to apply the strategy to.
        time_frame : str
            filepath to the data.
        start : str
            start date for data import.
        end : str
            end date for data import.
        tc : float
            proportional trading costs per trade.
        """
        self.strategy = strategy
        self.symbol = symbol
        self.time_frame = time_frame
        self.start = start
        self.end = end
        self.tc = tc
        self.ohlcv_data = self.get_data()
        self.position_data = None
        self.opt_results = None
        self.best_params = None

    def get_data(self) -> pd.DataFrame:
        """
        Imports the data.

        Returns
        -------
        pd.DataFrame
            OHLCV data within the specified time range.
        """
        dates = pd.date_range(start=pd.to_datetime(self.start).replace(day=1),
                              end=pd.to_datetime(self.end) + pd.DateOffset(months=1),
                              freq='MS')
        data = []
        for date in dates:
            date_ = date.strftime(format="%Y-%m")
            filepath = "https://data.binance.vision/data/spot/monthly/klines/" \
                       f"{self.symbol}/{self.time_frame}/" \
                       f"{self.symbol}-{self.time_frame}-{date_}.zip"
            try:
                data_date = pd.read_csv(
                    filepath,
                    usecols=range(6),
                    names=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
            except HTTPError:
                print(f"No data found in {filepath}.")
                continue

            data_date.index = pd.to_datetime(data_date['Timestamp'], unit='ms')
            data_date.index.name = "Datetime"
            data_date.drop(columns=["Timestamp"], inplace=True)
            data.append(data_date)

        raw = pd.concat(data, axis=0)
        return raw

    def test_strategy(self, parameters: dict) -> None:
        """
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ----------
        parameters : dict
            parameters for the strategy.
        """
        # Generate features and positions
        self.prepare_data(parameters=parameters)
        self.run_backtest()

        self.position_data["cum_returns"] = np.exp(self.position_data["returns"].cumsum())
        self.position_data["cum_strategy"] = np.exp(self.position_data["strategy"].cumsum())

        self.print_performance(parameters=parameters)

    def prepare_data(self, parameters: dict) -> None:
        """
        Prepares the Data for Backtesting.

        Parameters
        ----------
        parameters : dict
            parameters for the strategy.
        """
        features_data = self.strategy.generate_features(ohlcv_df=self.ohlcv_data)
        self.position_data = self.strategy.define_position(
            features_df=features_data,
            params=parameters
        )

    def run_backtest(self):
        """
        Runs the strategy backtest.
        """
        self.position_data["strategy"] = self.position_data["position"].shift() * self.position_data["returns"]
        self.position_data["trades"] = self.position_data["position"].diff().fillna(0).abs()
        self.position_data["strategy"] = self.position_data["strategy"] + self.position_data["trades"] * self.tc

    def plot_results(self):
        """
        Plots the cumulative performance of the trading strategy
        compared to buy-and-hold.
        """
        if self.position_data is None:
            print("Run test_strategy() first.")
        else:
            plt.figure()
            self.position_data[["cum_returns", "cum_strategy"]].plot(title=self.symbol,
                                                                     figsize=(12, 8))
            plt.show()

    def optimize_strategy(self, param_ranges: dict, metric: str = "Multiple"):
        """
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ============
        param_ranges: dict
            dict of ranges for the parameters.

        metric: str
            performance metric to be optimized (can be "Multiple" or "Sharpe")
        """

        if metric == "Multiple":
            performance_function = self.calculate_multiple
        else:
            raise ValueError(f"Selected metric: {metric} is not currently supported.")

        combinations = list(product(*param_ranges.values()))

        performance = []
        for comb in combinations:
            self.prepare_data(parameters=dict(zip(param_ranges.keys(), comb)))
            self.run_backtest()
            performance.append(performance_function(self.position_data["strategy"]))

        self.opt_results = pd.DataFrame(data=np.array(combinations), columns=param_ranges.keys())
        self.opt_results["performance"] = performance
        self.find_best_strategy()

    def find_best_strategy(self):
        """
        Finds the optimal strategy (global maximum).
        """
        best = self.opt_results.nlargest(1, "performance")
        self.best_params = best.drop(columns=["performance"]).iloc[0].to_dict()
        self.test_strategy(parameters=self.best_params)

    def print_performance(self, parameters: dict):
        """
        Calculates and prints various Performance Metrics.

        Parameters
        ----------
        parameters : dict
            parameters for the strategy.
        """
        strategy_multiple = round(self.calculate_multiple(self.position_data["strategy"]), 6)
        bh_multiple = round(self.calculate_multiple(self.position_data["returns"]), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(self.position_data["strategy"]), 6)

        print(100 * "=")
        print(f"SIMPLE PRICE & VOLUME STRATEGY | INSTRUMENT = {self.symbol} | PARAMETERS = {parameters}")
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print(100 * "=")

    @staticmethod
    def calculate_multiple(series):
        return np.exp(series.sum())

    @staticmethod
    def calculate_cagr(series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1
