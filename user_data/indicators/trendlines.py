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
import math
import timeit

# import operator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from labellines import labelLine, labelLines


# ZigZag

# This is inside your IPython Notebook
import pyximport
# pyximport.install()
pyximport.install(reload_support=True)
from user_data.indicators.zigzag_hi_lo import peak_valley_pivots as peak_valley_pivots_dynamic
from zigzag import peak_valley_pivots

from matplotlib.pyplot import plot, grid, show, savefig

import freqtrade.vendor.qtpylib.indicators as qtpylib


def plot_pivots(X, L, H, pivots):
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.2)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], H[pivots == 1], color='r', alpha=0.3)
    plt.scatter(np.arange(len(X))[pivots == -1], L[pivots == -1], color='g', alpha=0.3)
#     plt.show()
    pass

def plot_trends(df, interval: int, pair: str=None, filename: str=None):
    plt.figure(num=0, figsize=(20,10))
    df['old_date'] = df['date']
    to_datetime(df['date'])
    df.set_index(['date'],inplace=True)
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
        filename = 'chart_plots/' + pair.replace('/', '-') + '-' +  interval +  datetime.utcnow().strftime('-%H') + '.png'

    # plt.scatter(df.index[pivots == 1], df.high[pivots == 1], color='r')
    # plt.scatter(df.index[pivots == -1], df.low[pivots == -1], color='g')

    plt.savefig(filename)
    plt.close()
    df['date'] = df['old_date']

def plot_trends_new(df, interval: int, pair: str=None,  filename: str=None, type: str=None):
    plt.figure(num=0, figsize=(20,10))
    # df['old_date'] = df['date']
    # to_datetime(df['date'])
    # df.set_index(['date'],inplace=True)
    # plt.plot(df.index, df['res_trend'], 'r', label='resistance trend', linewidth=2)
    # plt.plot(df.index, df['sup_trend'], 'g', label='support trend', linewidth=2)
    plt.plot(df.val, 'k', alpha=0.5)
    plt.plot(df.index, df.sup_trend, 'r', label='res_trend', linewidth=1)
    plt.plot(df.index, df.sup_trend, 'g', label='sup_trend', linewidth=1)
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

def gentrends(self, serie: Series, interval: int, type: str, tolerance: int, confirmations: int, \
            angle_min: int, angle_max: int, thresh_up: int, thresh_down: int, chart=False):

    print ('------------------------------------------------------------------')
    print ('                       Generating Trendlines                      ')
    print ('------------------------------------------------------------------')

    print ('type: ', type)
    #new dataframe for work
    df = serie.to_frame(name='val')
    df['trend']= Series()
    # volatility settings for threshold for peak and valleys
    volat_window = {
                '1d':10,
                '4h':20,
                '1h':50,
                '30m':60,
                '15m':80,
                '5m':100,
                '1m':200
                }

    window = volat_window[interval]
    # Bollinger bands
    bollinger = qtpylib.bollinger_bands(serie, window=40, stds=2)
    df['bb_lowerband'] = bollinger['lower']
    df['bb_upperband'] = bollinger['upper']

    # create the BB expansion indicator
    df['bb_exp'] = (df['bb_upperband'] - df['bb_lowerband']) / df['bb_upperband']

    # pivots_list = peak_valley_pivots(df.val.values, thresh_up, thresh_down)
    pivots_list = peak_valley_pivots_dynamic(df.val.values, df.bb_exp.values)
    df['pivots'] = np.transpose(np.array((pivots_list)))

    max_confirmations = 0

    # global df
    # df = list()

    # get only peaks of type we want
    if type == "res":
        p = df.loc[df['pivots']==1]
        # p = df.loc[(df.val.shift(2) <= df.val.shift(1)) &
        #         (df.val.shift(1) <= df.val) &
        #         (df.val.shift(-1) <= df.val) &
        #         (df.val.shift(-2) <= df.val.shift(-1))]
        # id_max = p[:-1].high.values.argmax()
    if type == "sup":
        p = df.loc[df['pivots']==-1]
        # p = df.loc[(df.val.shift(2) >= df.val.shift(1)) &
        #         (df.val.shift(1) >= df.val) &
        #         (df.val.shift(-1) >= df.val) &
        #         (df.val.shift(-2) >= df.val.shift(-1))]
        # id_min = p[:-1].low.values.argmin()

    # print ('len DF: ', len(df))
    # print ('highs: ', len(h))
    # print ('lows: ', len(l))

    # def foo(row):
    #     return pandas.Series({"X": row["A"]+row["B"], "Y": row["A"]-row["B"]})
    #
    # df.apply(foo, axis=1)

    # find all trends
    print ('df: ', len(df))
    print ('pivots: ', len(p))
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

            df.loc[ax:,trend_name] = trend

            cond = True

            # print (angle)
            if type == "res":
                cond = by > df.loc[bx][trend_name]
                # id_max = p[:-1].high.values.argmax()
            if type == "sup":
                cond = by < df.loc[bx][trend_name]
            # if a high/low is higher/lower than current trend, it converts in the new trend
            if cond and (angle_max >= angle >= angle_min):
#                 print (len(df.val[ax:]), ' / ', len(trend))
#                 print (df.val[ax:])
#                 print (trend)
                trend_confirmations = df.loc[in_range(df['val'][ax:].values,trend, tolerance)]
                num_conf = len(trend_confirmations.index)
                if num_conf > max_confirmations:
                    max_confirmations = num_conf
                if num_conf >= confirmations:
                    # print ('max-confirmations: ', max_confirmations)
                    #first confirmed point in trend
                    # print('confirmations: ', len(trend_confirmations))
#                     print (trend_confirmations.index[confirmations - 1])

                    bxconf = df.index[trend_confirmations.index[confirmations - 1]]
#                     print ('conf: ', bxconf)
                    tconf = df.index[bxconf:]
                    trend = polyval([slope,intercept],tconf)
#                     print (trend_confirmations)
                    df.loc[:(bxconf-1),trend_name] = 0
                    df.loc[bxconf:,trend_name] = trend
                    # print (len(trend_confirmations))
#                     print (angle)

#                     if type == "res":
#                         cond = by > df.loc[bx][trend_name]
#                         # id_max = p[:-1].high.values.argmax()
#                     if type == "sup":
#                         cond = by < df.loc[bx][trend_name]
                else:
                    # print('deleting trend: ', trend_name)
                    del df[trend_name]
            # if a high/low is higher/lower than current trend, it converts in the new trend
#             if cond:
#                 t = df.index[p.index[i]:]
#                 slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
#                 # print (angle)
#                 # slope_angle = math.atan(slope)  # slope angle in radians
#                 # print (slope_angle, 'radians')
#                 # slope_angle_degrees = math.degrees(slope_angle)  # slope angle in degrees
#                 # print (slope_angle_degrees, 'grados')
#                 trend_next_wave = polyval([slope,intercept],t)
#                 angle = np.rad2deg(np.arctan2(trend_next_wave[-1] - trend_next_wave[0], t[-1] - t[0]))
#                 trend_name = 'trend|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)

#                 if angle > angle_min and angle < angle_max:
#                     trend_confirmations = df.loc[in_range(t,trend_next_wave, tolerance)]
#                     if len(trend_confirmations) >= confirmations:
#                         # print (len(trend_confirmations))
#                         print (angle)
#                         df.loc[ax:,trend_name] = trend_next_wave

                # if current_price < df.loc[-1][trend_name] && trend_confirmations >= min_tests:
                #     trend = {'name':trend_name,
                #             'interval':interval,
                #             'a':[ax, ay, df.iloc[ax].date],
                #             'b':[bx, by, df.iloc[bx].date],
                #             'slope':slope,
                #             'conf_n':trend_confirmations,
                #             'type':'res',
                #             'last':False,
                #             'max':False,
                #             'min':False}
                #     trends.append(trend)

    trends_names = [c for c in df if c.startswith('trend|')]
    print ('trend count: ', len(trends_names))
    print ('max-confirmations: ', max_confirmations)
    print('pivots: ', len(df.loc[(df.pivots == -1) |(df.pivots == 1)]))
    print ('setting sup/res')
    # print (mask)
    trends_column_names = [c for c in df if c.startswith('trend|')]
    if type == "res":
        def set_res(row):
            # print ('df.apply')
            # for t in row[[c for c in df if c.startswith('trend')]]:

            lower_res = 9999999999999
            # print (row)
            for t in trends_column_names:
                if row[t] != 0 and row[t] > row.val and row[t] < lower_res:
                    lower_res = row[t]
                    # print (lower_res)
            if lower_res > 0:
                # print ('higher', lower_res)
                # row['trend'] = lower_res
                return lower_res
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
                if row[t] > 0 and row[t] < row.val and row[t] > higher_sup:
                    higher_sup = row[t]
                    # print ('higher sup: ', higher_sup)
            if higher_sup > 0:
                # print ('higher', higher_sup)
                # row['trend'] = higher_sup
                return higher_sup

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
    # if chart == True:
    #     # plot_pivots(df.close, df.low, df.high, pivots)
    #     plot_trends_new(df, interval=interval, type=type)
    # print (df['trend'])
    return df['trend'], trends
