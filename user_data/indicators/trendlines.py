"""
Trend finder from dataframe
author@: Bruno Sarlo
github@: https://github.com/botum/trendline-indicator
"""

import copy
import sys
from typing import Dict, List, Tuple
from pandas import DataFrame, to_datetime, Series
import numpy as np

from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot, grid, show, savefig

from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from pylab import plot, title, show , legend

from technical import indicators

# from matplotlib.ticker import AutoMinorLocator

# ZigZag

# This is inside your IPython Notebook
import pyximport
# pyximport.install()
pyximport.install(reload_support=True)
from user_data.indicators import zigzag_hi_lo
from zigzag import *


def plot_pivots(X, pivots, interval: int, pair: str=None, filename: str=None):
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.2)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='r', alpha=0.3)
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='g', alpha=0.3)

    if not filename:
        filename = 'chart_plots/' + pair.replace('/', '-') + '-' +  interval +  datetime.utcnow().strftime('-%H') + '.png'

    # plt.scatter(df.index[pivots == 1], df.high[pivots == 1], color='r')
    # plt.scatter(df.index[pivots == -1], df.low[pivots == -1], color='g')

    plt.savefig(filename)
    plt.close()

# def plot_trends(df, interval: int, pivots, filename: str=None):
#     plt.figure(num=0, figsize=(20,10))
#     df['old_date'] = df['date']
#     to_datetime(df['date'])
#     df.set_index(['date'],inplace=True)
#     plt.plot(df.index, df['trendline-max'], 'r', label='resistance trend', linewidth=2)
#     plt.plot(df.index, df['trendline-min'], 'g', label='support trend', linewidth=2)
#     plt.plot(df.high, 'r', alpha=0.5)
#     plt.plot(df.close, 'k', alpha=0.5)
#     plt.plot(df.low, 'g', alpha=0.5)
#
#     plt.plot(df.bb_lowerband, 'b', alpha=0.5, linewidth=2)
#     plt.plot(df.bb_upperband, 'b', alpha=0.5, linewidth=2)
#
#     trends = [col for col in df if col.startswith('trend-')]
#     for t in trends:
#         plt.plot(df.index, df[t], 'k', label='trend', alpha=0.5, linewidth=1)
#
#     plt.xlim(df.index[0], df.index[-1])
#     plt.ylim(df.low.min()*0.99, df.high.max()*1.01)
#     plt.xticks(rotation='vertical')
#
#     if not filename:
#         filename = 'chart_plots/' +  interval + '-' +  datetime.utcnow().strftime('-%S') + '.png'
#
#     plt.scatter(df.index[pivots == 1], df.high[pivots == 1], color='r')
#     plt.scatter(df.index[pivots == -1], df.low[pivots == -1], color='g')
#
#     plt.savefig(filename)
#     plt.close()
#     df['date'] = df['old_date']


def plot_trends(df, interval: int, pair: str=None, filename: str=None):
    plt.figure(num=0, figsize=(20,10))
    # df['old_date'] = df['date']
    # to_datetime(df['date'])
    # df.set_index(['date'],inplace=True)
    plt.scatter(df.index, df['res_trend'], color='r', label='resistance trend', s=10)
    plt.scatter(df.index, df['sup_trend'], color='g', label='support trend', s=10)
    plt.plot(df.high, 'r', alpha=0.5)
    plt.plot(df.close, 'k', alpha=0.5)
    plt.plot(df.low, 'g', alpha=0.5)

    # plt.plot(df.bb_lowerband, 'b', alpha=0.5, linewidth=2)
    # plt.plot(df.bb_upperband, 'b', alpha=0.5, linewidth=2)

    trends = [col for col in df if col.startswith('trend|')]
    for t in trends:
        plt.plot(df.index, df[t], 'k', label='trend', alpha=0.5, linewidth=1)

    plt.xlim(df.index[0], df.index[-1])
    plt.ylim(df.low.min()*0.99, df.high.max()*1.01)
    plt.xticks(rotation='vertical')

    if not filename:
        filename = 'chart_plots/' + pair.replace('/', '-') + '-' +  interval + \
                str(len(df)) + datetime.utcnow().strftime('-%H') + '.png'

    # plt.scatter(df.index[pivots == 1], df.high[pivots == 1], color='r')
    # plt.scatter(df.index[pivots == -1], df.low[pivots == -1], color='g')

    plt.savefig(filename)
    plt.close()
    # df['date'] = df['old_date']

def plot_trends_new(df, interval: int, pair: str=None,  filename: str=None, type: str=None):
    plt.figure(num=0, figsize=(20,10))
    # df['old_date'] = df['date']
    # to_datetime(df['date'])
    # df.set_index(['date'],inplace=True)
    # plt.plot(df.index, df['res_trend'], 'r', label='resistance trend', linewidth=2)
    # plt.plot(df.index, df['sup_trend'], 'g', label='support trend', linewidth=2)
    plt.plot(df.val, 'k', alpha=0.5)
    plt.plot(df.index, df['res_trend'], 'r', label='res_trend', linewidth=1)
    plt.plot(df.index, df['sup_trend'], 'g', label='sup_trend', linewidth=1)
    trends = [col for col in df if col.startswith('trend|')]
    for t in trends:
        plt.plot(df.index, df[t], 'k', label=t, alpha=0.1, linewidth=1)
        # plt.text(df.index, df[t], t, fontsize=10)

    plt.xlim(df.index[0], df.index[-1])
    plt.ylim(df.val.min()*0.99, df.val.max()*1.01)
    plt.xticks(rotation='vertical')

    if not filename:
        filename = 'chart_plots/' + pair.replace('/', '-') +  interval + '-' + type + '.png'

    plt.scatter(df.loc[pivots == 1].index, df.loc[pivots == 1].high, color='r')
    plt.scatter(df.loc[pivots == -1].index, df.loc[pivots == -1].low, color='g')
    # labelLines(plt.gca().get_lines(),zorder=2.5)

    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    art.append(lgd)

    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig(filename, additional_artists=art, bbox_inches="tight")
    plt.close()
    # df['date'] = df['old_date']

def in_range(x, target, percent):
   start = target - target * percent
   end = target + target * percent
   check = (start <= x) & (end >= x)
   # print (check)
   return check

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

def get_confirmations(df, trend_name, pt, first):

    trend = df[trend_name]

    if pt == 'res':
        tolerance = 0.0001
        t_r = df.loc[(df['pivots']==1) & in_range(df['high'],trend, tolerance)]
    if pt == 'sup':
        tolerance = 0.0001
        t_r = df.loc[(df['pivots']==-1 ) & in_range(df['low'],trend, tolerance)]

    trend_confirmations = len(t_r)
#     trends['trend'].append(trend)
#     print(trend_name)

    return trend_confirmations


def get_trends_serie(self, serie: Series, interval: int, type: str, tolerance: int, confirmations: int, ticker_gap: int, \
             fake: int, angle_min: int, angle_max: int, thresh_up: int, thresh_down: int, chart=False, pair: str=None):
    df = serie.to_frame(name='val')
    df['trend']= Series()
    # print (pair)
    # print (type)
    # print (len(df))
    # volatility settings for threshold for peak and valleys
    volat_window = {
                '1d':10,
                '4h':20,
                '1h':50,
                '30m':20,
                '15m':30,
                '5m':50,
                '1m':100
                }

    window = volat_window[interval]
    # Bollinger bands
    bollinger = indicators.bollinger_bands(df, field='val', period=window, stdv=1.5)

    # create the BB expansion indicator
    df['bb_exp'] = (df['bb_upper'] - df['bb_lower']) / df['bb_upper']

    pivots_list = zigzag_hi_lo.peak_valley_pivots(df.val.values, df.val.values, df.bb_exp.values)

    # pivots_list = peak_valley_pivots(df.val.values, thresh_up, thresh_down)

    df['pivots'] = np.transpose(np.array((pivots_list)))
    # print (df['pivots'])
    max_confirmations = 0

    # global df
    # df = list()

    # get only peaks of type we want
    if type == "res":
        p = df.loc[df['pivots']==1]
    if type == "sup":
        p = df.loc[df['pivots']==-1]
    # print ('pivots: ', len(p))
    # print (p)
    for i in range(0, len(p)-1):
        ax = p.index[i]
        ay = p.iloc[i].val

        # trend x axis.
        t = df.index[ax:]

        # b point is next high
        # bx = p.index[i+1]
        # by = p.iloc[i+1].val
        # t = df.index[ax:]

        next_waves = p[i+1:]
        # if type == "res":
        #     cond = by > df.loc[bx][trend_name]
        #     # id_max = p[:-1].high.values.argmax()
        # if type == "sup":
        #     cond = by < df.loc[bx][trend_name]


        # check the rest of the highs if they are higher than trend
        for ib in range(0, len(next_waves)):
            # print (ib)
            bx = next_waves.index[ib]
            by = next_waves.iloc[ib].val

            # trace first trend
            slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
            trend = polyval([slope,intercept],t)
            angle = np.rad2deg(np.arctan2(by - ay, bx - ax))
            trend_name = 'trend|'+type+'|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
            # print (angle)
            if angle >= angle_min and angle <= angle_max:
                df.loc[ax:,trend_name] = trend
                # tests
                if type == "res":
                    # print ('min id: ', df['val'].rolling(window=ticker_gap).max().idxmin(), 'val', df.iloc[df['val'].rolling(window=ticker_gap).max().idxmin()][trend_name])
                    # print ('max id: ', df['val'].rolling(window=ticker_gap).max().idxmax(), 'val', df.iloc[df['val'].rolling(window=ticker_gap).max().idxmax()][trend_name])

                    tests = df.loc[in_range(df[trend_name], df['val'], tolerance) &
                    df['val'].rolling(window=ticker_gap).max() < \
                    df.iloc[df['val'].rolling(window=ticker_gap).max().idxmax()][trend_name]]
                    # print(tests)
                    # print('RESISTANCE MIN confirmations: ',confirmations)
                    # print('RESISTANCE found: ', len(tests))
                    if len(tests) > confirmations:
                        confirmed_idx = tests.index[confirmations-1]
                        # trace trend from confirmation point
                        # t_confirmed = df.index[confirmed_idx:]
                        df.loc[:confirmed_idx,trend_name] = np.NaN

                if type == "sup":
                    # print ('min id: ', df['val'].rolling(window=ticker_gap).min().idxmin(), 'val', df.iloc[df['val'].rolling(window=ticker_gap).min().idxmin()][trend_name])
                    # print ('max id: ', df['val'].rolling(window=ticker_gap).min().idxmax(), 'val', df.iloc[df['val'].rolling(window=ticker_gap).min().idxmax()][trend_name])
                    tests = df.loc[in_range(df[trend_name], df['val'], tolerance) &
                        df['val'].rolling(window=ticker_gap).min() > \
                        df.iloc[df['val'].rolling(window=ticker_gap).min().idxmin()][trend_name]]
                    # print(tests)
                    # print('SUPPORT MIN confirmations: ',confirmations)
                    # print('SUPPORT found: ', len(tests))
                    if len(tests) > confirmations:
                        confirmed_idx = tests.index[confirmations-1]
                        # trace trend from confirmation point
                        # t_confirmed = df.index[confirmed_idx:]
                        df.loc[:confirmed_idx,trend_name] = np.NaN
                        # slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                        # trend = polyval([slope,intercept],t_confirmed)
                        # del df[trend_name]
                        # df.loc[confirmed_idx:,trend_name] = trend
            # cond = True


    # trends_names = [c for c in df if c.startswith('trend|')]
    trends_column_names = [c for c in df if c.startswith('trend|')]
    if type == "res":
        def set_res(row):
            # print ('df.apply')
            # for t in row[[c for c in df if c.startswith('trend')]]:

            lower_res = 9999999999999
            # print (row)
            for t in trends_column_names:
                if row[t] != 0 and (row[t] * (1+fake)) > row.val and row[t] < lower_res:
                    lower_res = row[t]
                    # print (lower_res)
            if lower_res != 9999999999999:
                # print ('higher', lower_res)
                # row['trend'] = lower_res
                return lower_res
            else:
                # print (df[:row.name].val.max())
                return df[:row.name].val.max()
        # mask = (df.val.shift(1) <= df.val) & (df.val.shift(-1) <= df.val)
        # df['trend'] = df[mask].apply(set_res, axis=1)
        # print ('count fractals: ', len(df.loc[(df.val.shift(1) <= df.val) & (df.val.shift(-1) <= df.val)]))
        # df = df[(df.val.shift(1) <= df.val) & (df.val.shift(-1) <= df.val)].assign(trend=df.apply(set_res, axis=1))
        df = df.assign(trend=df.apply(set_res, axis=1))

    if type == "sup":
        def set_sup(row):
            # print ('df.apply')
            # for t in row[[c for c in df if c.startswith('trend')]]:
            higher_sup = 0
            for t in trends_column_names:
                if row[t] > 0 and (row[t] * (1-fake)) < row.val and row[t] > higher_sup:
                    higher_sup = row[t]
                    # print ('higher sup: ', higher_sup)
            if higher_sup > 0:
                # print ('higher', higher_sup)
                # row['trend'] = higher_sup
                return higher_sup
            else:
                # print (df[:row.name].val.min())
                prev_low = df[:row.name].val.min()
                if prev_low < row.val:
                    return df[:row.name].val.min()
                else:
                    return row.val * (1-tolerance)

        # mask = (df.val.shift(1) >= df.val) & (df.val.shift(-1) >= df.val)
        # df['trend'] = df[mask].apply(set_sup, axis=1)
        # df.trend.fillna(method='ffill')
        df = df.assign(trend=df.apply(set_sup, axis=1))
        # df = df[(df.val.shift(1) >= df.val) &
        #         (df.val.shift(-1) >= df.val)].assign(trend=df.apply(set_sup, axis=1))
        # df = df[(df.val.shift(2) <= df.val.shift(1)) &
        #         (df.val.shift(1) <= df.val) &
        #         (df.val.shift(-1) <= df.val) &
        #         (df.val.shift(-2) <= df.val.shift(-1))].assign(trend=df.apply(set_sup, axis=1))
        # df['trend'] = df[(df.val.shift(1) >= df.val) & (df.val.shift(-1) >= df.val)].apply(set_sup, axis=1)


    # print (df)
    trends = df[[c for c in df.columns if c.startswith('trend|')]]
    # print (trends)
    if chart == True:
        # plot_pivots(df.close, df.low, df.high, pivots)
        plot_pivots(df['val'], pivots_list, pair=pair, interval=interval)
    # print (df['trend'])
    return df['trend'], trends

def get_trends_lightbuoy_OHCL(self, df, interval: int, chart=False):

    # volatility settings for threshold for peak and valleys
    volat_window = {
                '5m':5,
                '1m':10
                }

    window = volat_window[interval]
    df['bb_exp'] = (df.bb_upperband.rolling(window=window).max() - df.bb_lowerband.rolling(window=window).min()) / df.bb_upperband.rolling(window=window).max()

    pivots = zigzag_hi_lo.peak_valley_pivots(df.low.values, df.high.values, df.bb_exp.values)
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
