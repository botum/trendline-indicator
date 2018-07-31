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
    stoploss = -0.03

    # Optimal ticker interval for the strategy
    ticker_interval = "1m"

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
            interval=self.ticker_interval, pivot_type='pivots',
            tolerance=0.00001, su_min_tests=4, re_min_tests=2, body_min_tests=1, ticker_gap = 5, fake=0.0001, nearby=0.001,
            angle_max = 100, angle_min = 80,
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

    def populate_buy_trend(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                (dataframe['close'] == dataframe.s1_trend)
                # |
                # (dataframe['close'] > dataframe.re_trend)
                # (dataframe['close'] <= dataframe.sup_trend)
                &
                (dataframe['volume'].shift(1) < dataframe['volume']/10)
                &
                (dataframe['rsi'] < 25)
                &
                (dataframe.r1_trend >= dataframe.s1_trend*1.03)
                # (dataframe['close']==dataframe['sup_trend'])
                # 0
            ),
            'buy'] = 1


        # UNCOMMENT TO PLOT
        # self.plot_dataframe(dataframe, pair, ['s1_trend,r1_trend', '', 'rsi'])

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
                (dataframe['close'] <= dataframe['s1_trend'] * 0.998)
                # |
                # (dataframe['close'] <= dataframe['sup_trend'] * (1 - dataframe.iloc[-1].bb_exp))
                # 0
            ),
            'sell'] = 1

        return dataframe


    def plot_dataframe(self, data, pair, indicators: list):
        """
        plots our dataframe everytime new data arrive
        and a tick is closed
        :param indicators: list of indicators
        :param data:
        :return:
        """

        import plotly.graph_objs as go
        from plotly import tools
        from plotly.offline import plot as plt

        def generate_row(fig, row, raw_indicators, data) -> tools.make_subplots:
            """
            Generator all the indicator selected by the user for a specific row
            """
            if raw_indicators is None or raw_indicators == "":
                return fig
            for indicator in raw_indicators.split(','):
                if indicator in data:
                    scattergl = go.Scattergl(
                        x=data['date'],
                        y=data[indicator],
                        name=indicator
                    )
                fig.append_trace(scattergl, row, 1)


            return fig

        rows = len(indicators)

        if rows < 3:
            rows = 3

        # Define the graph
        fig = tools.make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            row_width=[1, 1, 4],
            vertical_spacing=0.0001,
        )

        fig['layout'].update(title=pair)
        fig['layout']['yaxis1'].update(title='Price')
        fig['layout']['yaxis2'].update(title='Volume')
        fig['layout']['yaxis3'].update(title='Other')

        if rows > 3:
            for x in range(3, rows):
                fig['layout']['yaxis{}'.format(x)].update(title='Other {}'.format(x))

        # Common information
        candles = go.Candlestick(
            x=data.date,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            name='Price'
        )
        fig.append_trace(candles, 1, 1)

        df_buy = data[data['buy'] == 1]
        buys = go.Scattergl(
            x=df_buy.date,
            y=df_buy.close,
            mode='markers',
            name='buy',
            marker=dict(
                symbol='triangle-up-dot',
                size=9,
                line=dict(width=1),
                color='green',
            )
        )

        fig.append_trace(buys, 1, 1)
        # Row 2
        volume = go.Bar(
            x=data['date'],
            y=data['volume'],
            name='Volume'
        )
        fig.append_trace(volume, 2, 1)

        row = 0
        for indicator in indicators:
            row = row + 1
            # print(row)
            generate_row(fig, row, indicator, data)

        from pathlib import Path
        plt(fig, auto_open=False, filename=str(
            Path('user_data').joinpath(
                "{}_{}_analyze_{}_{}.html".format(__class__.__name__, pair.replace('/', '-'), self.ticker_interval, data['date'].iloc[-1]))))
