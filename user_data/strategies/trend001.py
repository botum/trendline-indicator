# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
# from scripts import trendy
from indicators import in_range, went_down, get_trend_lines, get_pivots
# from freqtrade.persistence import Pair

# from freqtrade.persistence import *

from user_data.indicators.trendlines import *

class_name = 'DefaultStrategy'

# pair = 'ETH/BTC'

class trend001(IStrategy):

    """

    author@: Bruno Sarlo

    Strategy for buying the trendline

    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "1200":  0.000,
        "600":  0.04,
        "0":  0.05
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.03

    # Optimal ticker interval for the strategy
    ticker_interval = "5m"

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:

        """
        Indicator for trends
        """

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # dataframe = get_pivots(self, dataframe)
        dataframe = gentrends(self, dataframe, self.ticker_interval, chart=True)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                (in_range(dataframe['close'],dataframe['trendline-min']*1.001, 0.001))
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['trendline-max']*0.99)
                |
                (dataframe['close'] <= dataframe['trendline-min'] * 0.95)

            ),
            'sell'] = 1
        # print (dataframe.loc[dataframe['sell']==1].close)

        return dataframe



    def did_bought(self):
        """
        we are notified that a given pair was bought
        :param pair: the pair that was is concerned by the dataframe
        """

    def did_sold(self):
        """
        we are notified that a given pair was sold
        :param pair: the pair that was is concerned by the dataframe
        """

    def did_cancel_buy(self):
        """
        we are notified that a given pair buy was not filled
        :param pair: the pair that was is concerned by the dataframe
        """

    def did_cancel_sell(self):
        """
        we are notified that a given pair was not sold
        :param pair: the pair that was is concerned by the dataframe
        """
