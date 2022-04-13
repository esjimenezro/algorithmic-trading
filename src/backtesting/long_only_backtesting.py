from itertools import product
import pandas as pd
import numpy as np


class LongOnlyBacktester:
    """ Class for the vectorized backtesting of simple Long-only trading strategies.

    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade


    Methods
    =======
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

    def __init__(self, filepath, symbol, start, end, tc):

        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        self.data = None

    def __repr__(self):
        return "Long_Only_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    def get_data(self):
        """ Imports the data.
        """
        raw = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw

    def test_strategy(self, percentiles=None, thresh=None):
        """
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        percentiles: tuple (return_perc, vol_low_perc, vol_high_perc)
            return and volume percentiles to be considered for the strategy.

        thresh: tuple (return_thresh, vol_low_thresh, vol_high_thesh)
            return and volume thresholds to be considered for the strategy.
        """

        self.prepare_data(percentiles=percentiles, thresh=thresh)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        self.print_performance()

    def prepare_data(self, percentiles, thresh):
        """ Prepares the Data for Backtesting.
        """

        data = self.data[["Close", "Volume", "returns"]].copy()
        data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
        data.loc[data.vol_ch < -3, "vol_ch"] = np.nan

        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), percentiles[0])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[1], percentiles[2]])
        elif thresh:
            self.return_thresh = thresh[0]
            self.volume_thresh = [thresh[1], thresh[2]]

        cond1 = data.returns >= self.return_thresh
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])

        data["position"] = 1
        data.loc[cond1 & cond2, "position"] = 0

        self.results = data

    def run_backtest(self):
        """ Runs the strategy backtest.
        """

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc

        self.results = data

    def plot_results(self):
        """  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        """
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    def optimize_strategy(self, return_range, vol_low_range, vol_high_range, metric="Multiple"):
        """
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ============
        return_range: tuple
            tuples of the form (start, end, step size).

        vol_low_range: tuple
            tuples of the form (start, end, step size).

        vol_high_range: tuple
            tuples of the form (start, end, step size).

        metric: str
            performance metric to be optimized (can be "Multiple" or "Sharpe")
        """

        self.metric = metric

        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe

        return_range = range(*return_range)
        vol_low_range = range(*vol_low_range)
        vol_high_range = range(*vol_high_range)

        combinations = list(product(return_range, vol_low_range, vol_high_range))

        performance = []
        for comb in combinations:
            self.prepare_data(percentiles=comb, thresh=None)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))

        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["returns", "vol_low", "vol_high"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()

    def find_best_strategy(self):
        """ Finds the optimal strategy (global maximum).
        """

        best = self.results_overview.nlargest(1, "performance")
        return_perc = best.returns.iloc[0]
        vol_perc = [best.vol_low.iloc[0], best.vol_high.iloc[0]]
        perf = best.performance.iloc[0]
        print("Return_Perc: {} | Volume_Perc: {} | {}: {}".format(return_perc, vol_perc, self.metric, round(perf, 5)))
        self.test_strategy(percentiles=(return_perc, vol_perc[0], vol_perc[1]))

    def print_performance(self):
        """ Calculates and prints various Performance Metrics.
        """

        data = self.results.copy()
        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(data.strategy), 6)
        ann_mean = round(self.calculate_annualized_mean(data.strategy), 6)
        ann_std = round(self.calculate_annualized_std(data.strategy), 6)
        sharpe = round(self.calculate_sharpe(data.strategy), 6)

        print(100 * "=")
        print("SIMPLE PRICE & VOLUME STRATEGY | INSTRUMENT = {} | THRESHOLDS = {}, {}".format(self.symbol, np.round(
            self.return_thresh, 5), np.round(self.volume_thresh, 5)))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))

        print(100 * "=")

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
