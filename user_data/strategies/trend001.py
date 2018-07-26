# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
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
        "120":  0.00001,
        "80":  0.01,
        "60":  0.001,
        "30":  0.005,
        "0":  0.01
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.03

    # Optimal ticker interval for the strategy
    ticker_interval = "5m"

    def populate_indicators(self, dataframe: DataFrame, pair: str) -> DataFrame:

        """
        Indicator for trends
        """


        # dataframe = get_pivots(self, dataframe)
        # print (len(dataframe), 'before get_trends')
        # dataframe = get_trends(self, dataframe.high,
        #             interval=self.ticker_interval,
        #             type='res', tolerance=0.001, confirmations=3,
        #             slope_max = 20,
        #             slope_min = 95,
        #             chart=False)

        # dataframe['res_trend'] = get_trends(self, dataframe.high,
        #             interval=self.ticker_interval,
        #             type='res', tolerance=0.001, confirmations=3,
        #             angle_max = 20, angle_min = 95,
        #             thresh_up = 0.01, thresh_down = -0.01,
        #             chart=True)
        dataframe['res_trend'], res_trends = get_trends_serie(self, dataframe.high,
                    interval=self.ticker_interval,
                    type='res', tolerance=0.0001, confirmations=2,
                    angle_max = 90,
                    angle_min = -70,
                    thresh_up = 0.02, thresh_down = -0.02,
                    chart=False)
        dataframe['sup_trend'], sup_trends = get_trends_serie(self, dataframe.low,
                    interval=self.ticker_interval,
                    type='sup', tolerance=0.0001, confirmations=3,
                    angle_max = 180,
                    angle_min = 0,
                    thresh_up = 0.02, thresh_down = -0.02,
                    chart=False)


        # print (dataframe.sup_trend)
        # plot_trends_new(dataframe, interval=self.ticker_interval)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                (in_range(dataframe['close'],dataframe.sup_trend*1.001, 0.001))
                # (dataframe['close']==dataframe['sup_trend'])
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
                # (dataframe['close'] >= dataframe['res_trend'].shift(2))
                # |
                # (dataframe['close'] <= dataframe['sup_trend'] * 0.95)
                0
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
