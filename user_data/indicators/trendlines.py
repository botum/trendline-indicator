"""
Trend finder from dataframe
author@: Bruno Sarlo
github@: https://github.com/botum/trendline-indicator
"""

import copy
import sys
from typing import Dict, List, Tuple
from pandas import DataFrame, to_datetime, Series

from datetime import datetime


from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from pylab import plot, title, show , legend

from freqtrade import (DependencyException, OperationalException, exchange, persistence)
from freqtrade.configuration import Configuration
# from freqtrade.persistence import Trade, Pair, Trend
from freqtrade.indicators import in_range
# import operator

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates

# ZigZag

# This is inside your IPython Notebook
import pyximport
# pyximport.install()
pyximport.install(reload_support=True)
from user_data.indicators.zigzag_hi_lo import *

from matplotlib.pyplot import plot, grid, show, savefig

def plot_pivots(X, L, H, pivots):
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.2)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], H[pivots == 1], color='r', alpha=0.3)
    plt.scatter(np.arange(len(X))[pivots == -1], L[pivots == -1], color='g', alpha=0.3)
#     plt.show()
    pass

def plot_trends(df, interval: int, pivots, filename: str=None):
    plt.figure(num=0, figsize=(20,10))
    df['old_date'] = df['date']
    to_datetime(df['date'])
    df.set_index(['date'],inplace=True)
    plt.plot(df.index, df['trendline-max'], 'r', label='resistance trend', linewidth=2)
    plt.plot(df.index, df['trendline-min'], 'g', label='support trend', linewidth=2)
    plt.plot(df.high, 'r', alpha=0.5)
    plt.plot(df.close, 'k', alpha=0.5)
    plt.plot(df.low, 'g', alpha=0.5)

    plt.plot(df.bb_lowerband, 'b', alpha=0.5, linewidth=2)
    plt.plot(df.bb_upperband, 'b', alpha=0.5, linewidth=2)

    trends = [col for col in df if col.startswith('trend-')]
    for t in trends:
        plt.plot(df.index, df[t], 'k', label='trend', alpha=0.5, linewidth=1)

    plt.xlim(df.index[0], df.index[-1])
    plt.ylim(df.low.min()*0.99, df.high.max()*1.01)
    plt.xticks(rotation='vertical')

    if not filename:
        filename = 'chart_plots/' +  interval + '-' +  datetime.utcnow().strftime('-%S') + '.png'

    plt.scatter(df.index[pivots == 1], df.high[pivots == 1], color='r')
    plt.scatter(df.index[pivots == -1], df.low[pivots == -1], color='g')

    plt.savefig(filename)
    plt.close()
    df['date'] = df['old_date']

def get_tests(df, trend_name, pt, first):

    trend = df[trend_name]

    if pt == 'res':
        tolerance = 0.0001
        t_r = df.loc[(df['pivots']==1) & in_range(df['high'],trend, tolerance)]
    if pt == 'sup':
        tolerance = 0.0001
        t_r = df.loc[(df['pivots']==-1 ) & in_range(df['low'],trend, tolerance)]

    trend_tests = len(t_r)
#     trends['trend'].append(trend)
#     print(trend_name)

    return trend_tests

def gentrends(self, df, interval: int, chart=False):

    # volatility settings for threshold for peak and valleys
    volat_window = {
                '5m':5,
                '1m':10
                }

    window = volat_window[interval]
    df['bb_exp'] = (df.bb_upperband.rolling(window=window).max() - df.bb_lowerband.rolling(window=window).min()) / df.bb_upperband.rolling(window=window).max()

    pivots = peak_valley_pivots(df.low.values, df.high.values, df.bb_exp.values)
    df['pivots'] = np.transpose(np.array((pivots)))

    # separate high and low peaks
    h = df.loc[df['pivots']==1]
    l = df.loc[df['pivots']==-1]


    df_orig = df

    # print ('len DF: ', len(df))
    # print ('highs: ', len(h))
    # print ('lows: ', len(l))

    global trends
    trends = list()

    # we get the highest and lowest
    id_max = h[:-1].high.values.argmax()
    id_min = l[:-1].low.values.argmin()

    # for each high...
    for i in range(0, len(h) -1):
        ax = h.index[i]
        ay = h.iloc[i].high

        # b point is next high
        bx = h.index[i+1]
        by = h.iloc[i+1].high
        t = df.index[ax:]

        # trace first trend
        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend_name = 't_r|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
        trend = polyval([slope,intercept],t)

        # plot trend to dataframe
        df.loc[h.index[i]:,trend_name] = trend

        next_waves = h[i+2:]
        is_last = len(next_waves[i:])==1
        trend_tests = get_tests(df, trend_name, 'res', True)

        # add first trend
        trend = {'name':trend_name,
                'interval':interval,
                'a':[ax, ay, df.iloc[ax].date],
                'b':[bx, by, df.iloc[bx].date],
                'slope':slope,
                'conf_n':trend_tests,
                'type':'res',
                'last':False,
                'max':False,
                'min':False}
        trends.append(trend)


        # check the rest of the highs if they are higher than trend
        for ib in range(0, len(next_waves)):
            bx = next_waves.index[ib]
            by = next_waves.iloc[ib].high

            # if a high is higher than current trend, it converts in the new trend
            if by > df.loc[bx][trend_name]:
                t = df.index[h.index[i]:]
                slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                trend_next_wave = polyval([slope,intercept],t)
                trend_name = 'trend-|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
                df.loc[h.index[i]:,trend_name] = trend_next_wave
                trend_tests = get_tests(df, trend_name, 'res', False)
                trend = {'name':trend_name,
                        'interval':interval,
                        'a':[ax, ay, df.iloc[ax].date],
                        'b':[bx, by, df.iloc[bx].date],
                        'slope':slope,
                        'conf_n':trend_tests,
                        'type':'res',
                        'last':False,
                        'max':False,
                        'min':False}
                trends.append(trend)

        trends[-1]['last'] = True

        if i == id_max:
            df_orig['trendline-max'] = df[trend_name]
            trends[-1]['max'] = True

    for i in range(0, len(l) -1):

        ax = l.index[i]
        ay = l.iloc[i].low

        bx = l.index[i+1]
        by = l.iloc[i+1].low
        t = df.index[ax:]

        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend_l = polyval([slope,intercept],t)

        trend_name = 't_s|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
        df.loc[l.index[i]:,trend_name] = trend_l

        next_waves = l[i+2:]
        is_last = len(next_waves[i:])==1
        trend_tests = get_tests(df, trend_name, 'sup', True)


        trend = {
            'name':trend_name,
            'interval':interval,
            'a':[ax, ay, df.iloc[ax].date],
            'b':[bx, by, df.iloc[bx].date],
            'slope':slope,
            'conf_n':trend_tests,
            'type':'sup',
            'last':False,
            'max':False,
            'min':False}
        trends.append(trend)

        for ib in range(0, len(next_waves)):
            bx = next_waves.index[ib]
            by = next_waves.iloc[ib].low
            if by < df.loc[bx][trend_name]:
                t = df.index[l.index[i]:]
                slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                trend_next_wave = polyval([slope,intercept],t)
                trend_name = 'trend-|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
                df.loc[l.index[i]:,trend_name] = trend_next_wave
                trend_tests = get_tests(df, trend_name, 'sup', False)
                trend = {
                    'name':trend_name,
                    'interval':interval,
                    'a':[ax, ay, df.iloc[ax].date],
                    'b':[bx, by, df.iloc[bx].date],
                    'slope':slope,
                    'conf_n':trend_tests,
                    'type':'sup',
                    'last':False,
                    'max':False,
                    'min':False}
                trends.append(trend)

        trends[-1]['last'] = True

        if i == id_min:
            df_orig['trendline-min'] = df[trend_name]
            trends[-1]['min'] = True
    if chart == True:
        # plot_pivots(df.close, df.low, df.high, pivots)
        plot_trends(df_orig, interval=interval, pivots=pivots)
    return df_orig
