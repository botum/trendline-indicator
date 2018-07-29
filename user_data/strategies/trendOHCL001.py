# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from user_data.indicators.trendlines import *
# from user_data.indicators.sure import *

class_name = 'DefaultStrategy'

# pair = 'ETH/BTC'

class trendOHCL001(IStrategy):

    """

    author@: Bruno Sarlo

    Strategy for buying and selling the trendlines

    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
        # "40":  0.00001,
        # "30":  0.002,
        # "60":  0.005,
        # "30":  0.02,
        "0":  0.03
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.01

    # Optimal ticker interval for the strategy
    ticker_interval = "5m"

    def populate_indicators(self, dataframe: DataFrame, pair: str) -> DataFrame:

        """
        Indicator for trends
        """

        #
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # dataframe['rsi_sup_trend'], rsi_sup_trends = get_trends_serie(self, dataframe['rsi'].fillna(100),
        #             interval=self.ticker_interval,
        #             type='sup', tolerance=0.0000001, min_tests=4,
        #             angle_max = 180,
        #             angle_min = -180,
        #             thresh_up = 0.5, thresh_down = -0.5,
        #             chart=True, pair=pair)


        # Bollinger bands
        # bollinger = indicators.bollinger_bands(dataframe, field='close', period=20, stdv=1.5)

        # create the BB expansion indicator
        # dataframe['bb_exp'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_upper']

        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        # dataframe['cci'] = ta.CCI(dataframe)
        #
        # dataframe['mfi'] = ta.MFI(dataframe)

        # dataframe['vol_res_trend'], rsi_sup_trends = get_trends_serie(self, dataframe['volume'],
        #             interval=self.ticker_interval,
        #             type='res', tolerance=0.0000001, min_tests=1,
        #             angle_max = 180,
        #             angle_min = -180,
        #             thresh_up = 5, thresh_down = -5,
        #             chart=True)
        # print (dataframe['rsi_res_trend'])
        # dataframe['rsi_sup_trend'], rsi_sup_trends = get_trends_serie(self, dataframe['rsi'],
        #             interval=self.ticker_interval,
        #             type='sup', tolerance=0.0001, min_tests=8,
        #             angle_max = 180,
        #             angle_min = 0,
        #             thresh_up = 0.008, thresh_down = -0.008,
        #             chart=False)
        # dataframe['cmf'] = cmf(dataframe)

        dataframe = get_trends_lightbuoy_OHCL(dataframe,
            interval=self.ticker_interval, pivot_type='fractals',
            tolerance=0.00001, su_min_tests=3, re_min_tests=5, body_min_tests=1, ticker_gap = 5, fake=0.00001, nearby=0.00001,
            angle_max = 85, angle_min = 75,
            thresh_up = 0.01, thresh_down = -0.01,
            chart=False, pair=pair)

        # dataframe, su, re = get_sure_zigzag_OHCL(self, dataframe,
        #             intervals=[self.ticker_interval], quantile=0.03,
        #             up_thresh=0.005, down_thresh=0.005)
            # print (dataframe.sup_trend)
        # plot_trends(dataframe, interval=self.ticker_interval, pair=pair)
        # print ('pair: ', pair)
        # print ('close: ', dataframe.iloc[-1].close)
        # print ('bb expan: ', dataframe.iloc[-1].bb_exp * 100, '%')
        # print ('resistence: ', (dataframe.iloc[-1].res_trend / dataframe.iloc[-1].sup_trend) * 100)
        # print ('stoploss: ',  dataframe.iloc[-1].sup_trend * (1 - (dataframe.iloc[-1].res_trend / dataframe.iloc[-1].sup_trend)) * 100 /  dataframe.iloc[-1].close, '%')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                (dataframe['close'] <= dataframe.s1_trend)
                # |
                # (dataframe['close'] > dataframe.re_trend)
                # (dataframe['close'] <= dataframe.sup_trend)
                &
                (dataframe['volume'].shift(1) < dataframe['volume']/3)
                # &
                # (dataframe['rsi'] < 25)
                &
                (dataframe.r1_trend >= dataframe.s1_trend*1.03)
                # (dataframe['close']==dataframe['sup_trend'])
                # 0
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
                (dataframe['close'] >= dataframe['r1_trend'])
                |
                (dataframe['close'] <= dataframe['s1_trend'] * 0.98)
                # |
                # (dataframe['close'] <= dataframe['sup_trend'] * (1 - dataframe.iloc[-1].bb_exp))
                # 0
            ),
            'sell'] = 1

        return dataframe
