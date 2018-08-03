"""
Trend finder from dataframe
author@: Bruno Sarlo
github@: https://github.com/botum/trendline-indicator
"""

from scipy import linspace, polyval, polyfit, sqrt, stats, randn


from user_data.indicators.util import *
# from user_data.indicators.trendlines import *
from technical import indicators, util
import warnings
# warnings.filterwarnings("always")

# ZigZag

# This is inside your IPython Notebook
pyximport.install()
pyximport.install(reload_support=True)

from IPython.display import display, HTML

import os
import subprocess

import json

from pandas import DataFrame, Series, to_datetime, isna, notna
import numpy as np

import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_trends(df, interval: int, abn: int=0, pair: str=None, filename: str=None, cols=[], i: int=0):
    plt.figure(num=0, figsize=(20,10))
    # df['old_date'] = df['date']
    # to_datetime(df['date'])
    # df.set_index(['date'],inplace=True)
#     for c in cols:
#         plt.scatter(df.index, df[c], color='r', label=c, s=10)


    if 's1_trend' in df:
        plt.plot(df['s1_trend'], color='g', label='support trend')
    if 'r1_trend' in df:
        plt.plot(df['r1_trend'], color='r', label='res trend')
    plt.plot(df.high, 'r', alpha=0.5)
    plt.plot(df.close, 'k', alpha=0.5)
    plt.plot(df.low, 'g', alpha=0.5)

#     plt.plot(df.bb_lower, 'b', alpha=0.5, linewidth=2)
#     plt.plot(df.bb_upper, 'b', alpha=0.5, linewidth=2)

    trends = [col for col in df if col.startswith('trend|')]
    for t in trends:
        plt.plot(df.index, df[t], 'k', label='trend', alpha=0.2, linewidth=1)
#         data_name = 'data_'+t+'|su'
#         if np.count_nonzero(np.where(df[data_name]==1)) > 2:
#             plt.plot(df.index, df[t], 'k', label='trend', alpha=0.5, linewidth=1)

    plt.xlim(df.index[0], df.index[-1])
    plt.ylim(df.low.min()*0.99, df.high.max()*1.01)
    plt.xticks(rotation='vertical')

    if not filename:
        filename = 'chart_plots/' + pair.replace('/', '-') + '-' +  interval + \
                str(len(df)) + datetime.utcnow().strftime('-%H') + '_' + str(i) + '_.png'
    if 'fractals' in df:
        res_pivots = df['fractals'] == 1
        sup_pivots = df['fractals'] == -1

        plt.scatter(df.index[res_pivots], df.high[res_pivots], color='r', s=5)
        plt.scatter(df.index[sup_pivots], df.low[sup_pivots], color='g', s=5)
    if 'pivots' in df:
        res_pivots = df['pivots'] == 1
        sup_pivots = df['pivots'] == -1

        plt.scatter(df.index[res_pivots], df.high[res_pivots], color='k', s=30)
        plt.scatter(df.index[sup_pivots], df.low[sup_pivots], color='k', s=30)

    plt.scatter(df.index[abn[0]], df.low[abn[0]], color='r', s=60)
    plt.scatter(df.index[abn[1]], df.low[abn[1]], color='g', s=60)
    plt.scatter(df.index[abn[2]], df.low[abn[2]], color='b', s=60)
    os.chdir('/home/bruno/Documentos/work/bitcoins/traders/freqtrade/chart_plots/')
    plt.savefig(filename)
    # plt.show()
    plt.close()


def get_trends_OHCL(df: DataFrame, interval: int, pressision: int, pivot_type: str, \
                     su_min_tests: int, re_min_tests: int, body_min_tests: int, \
                     ticker_gap: int, fake: int, nearby: int, angle_min: int, \
                     angle_max: int, thresh_up: int, thresh_down: int, \
                     chart=False, pair: str=None):
    ''' this approach is based on concurrence of points, either low high or body of candle'''

    # start_time = time.time()
    print('finding trends by concurrence')

    if pivot_type == 'pivots':
        df, pivots = get_pivots_OHCL(df, period=20, stdv= 1.2,
                    interval=interval, thresh_type = 'dynamic',
                    thresh_up = 0.05, thresh_down = -0.05,
                    chart=False, pair=pair)
    elif pivot_type == 'fractals':
        df = get_fractals(df)

    p = df.loc[df[pivot_type]==-1]

    for i in range(0, len(p)-1):
        ax = p.index[i]
        ay = p.iloc[i].low


        next_waves = p[i+1:]

        for i in range(0, len(next_waves)):
            # print (i)
            bx = next_waves.index[i]
            by = next_waves.iloc[i].low

            t = df.index[bx:]

            # trace first trend
            slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
            trend = polyval([slope,intercept],t)
            angle = calculate_angle(ax, ay, bx, by)
            trend_name = 'trend|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)+'|'+str(round(angle))

            cond = angle >= angle_min and angle <= angle_max
            if cond:
                df.loc[bx:,trend_name] = trend
                df = get_line_tests(df, trend_name, pivot_type, nearby, fake)
                data_name = 'data_'+ trend_name

                re_name = data_name+'|re'
                su_name = data_name+'|su'
                body_name = data_name+'|body'

                if np.count_nonzero(np.where(df[su_name]==1)) >= su_min_tests:
#                     print (df.index[df[su_name]==1])
                    cx = df.index[df[su_name]==1][su_min_tests-1]
                    df.loc[ax:cx,trend_name] = np.nan
                else:
                    del df[trend_name]
                    del df[re_name]
                    del df[su_name]
                    del df[body_name]

    # print("--- %s seconds ---" % (time.time() - start_time))
    return df

def get_trends_lightbuoy_OHCL(df: DataFrame, interval: int, pressision: int, pivot_type: str, \
                     su_min_tests: int, re_min_tests: int, body_min_tests: int, \
                     ticker_gap: int, fake: int, nearby: int, angle_min: int, \
                     angle_max: int, thresh_up: int, thresh_down: int, \
                     chart=False, pair: str=None):

    ''' this approach focus on the lower/higher trend progressivelly. Currently plotting after 2nd point
    as the third is our validation and we might want to buy or sell if broken.

    '''

    # start_time = time.time()
    # print('finding trends - lightbuoy style')

    if pivot_type == 'pivots':
        df, pivots = get_pivots_OHCL(df, period=20, stdv= .8,
                    interval=interval, thresh_type = 'dynamic',
                    thresh_up = thresh_up, thresh_down = thresh_down,
                    chart=False, pair=pair)
    elif pivot_type == 'fractals':
        df = get_fractals(df)

    # df = get_fractals(df)

    trends = []
    '''structure:
    - confirmed_trends[{'ax': ax, 'ay':ay...}, {}, {}]
    '''
    trends = DataFrame()

    r = df.loc[df[pivot_type]==1]
    rx = r.index[0]
    ry = r.iloc[0].high
    df.loc[0:r.index[1], 'r1_trend'] = ry

    s = df.loc[df[pivot_type]==-1]

    # just working with supports now
    p = s
    last=len(p)-1
    df.loc[0:p.index[1],'s1_trend'] = p.iloc[0].low

    ax = p.index[0]
    ay = p.iloc[0].low
    lightbuoy = ax

    bx = p.index[1]
    by = p.iloc[1].low

    # right now to get point N we have to wait for the next height pivot point,
    # this could be done with fractals
    nx = p.index[2]
    ny = p.iloc[2].low
    i = 2

    # plot_trends(df, interval=interval, pair=pair, abn=nx)
    b=True
    while b:
        # trace first trend
        # print('trend: ')
        t = df.index[ax:]
        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend_serie = polyval([slope,intercept],t)
        angle = calculate_angle(ax, ay, bx, by)
        trend_name = 'trend|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)+'|'+str(round(angle))
        df.loc[ax:,trend_name] = trend_serie

        trends = trends.append({
                'name':trend_name,
                'interval':interval,
                'ax':ax,
                'ay':ay,
                'bx':bx,
                'by':by,
                'angle':angle,
                'su_test':2, # just so we know last totals
                're_test':0,
                'body_test':0
                }, ignore_index=True)
        trend_row = trends.iloc[-1]

        # print ('ax: ', ax)
        # print ('ay: ', ay)
        # print ('bx: ', bx)
        # print ('by: ', by)
        # print ('nx: ', nx)
        # print ('ny: ', ny)
        cond =  angle >= angle_min and angle <= angle_max

        if ny >= (df.loc[nx][trend_name] * 1-fake) and cond:
            # print('going up')
            if in_range(ny, df.loc[nx][trend_name] * 1-fake, pressision):
                tests = trends.at[trends.index[-1], 'su_test'] = trend_row['su_test'] + 1
            # prev_ai.append(p.index[i+1])
                # else:
                # cond =  angle >= angle_min and angle <= angle_max \
                #         and trend['su_test'] >= su_min_tests
                #         # and (df.iloc[nx][trend_name]) >= df.iloc[nx]['low']
            if tests >= su_min_tests and cond:
                df.loc[nx:,'s1_trend'] = df.loc[nx:,trend_name]

            # df.loc[ax:,'s1_trend'] = df.loc[ax:,trend_name]
            # plot_trends(df, interval=interval, pair=pair, abn=[ax, bx, nx], i=i, filename='temp_plot'+str(i)+'.png')
            ax = bx
            ay = by
            bx = nx
            by = ny
            # print (trend_row)
        else:
            # print('going down')
            found_su = False
            # if cond == True and (df.iloc[nx][trend_name]) <= df.iloc[nx]['low'] and angle > 90:
            #     df.loc[nx:,'s1_trend'] = df.loc[nx:,trend_name]
            if in_range(df.loc[nx]['high'], df.loc[nx][trend_name], pressision):
                trends.at[trends.index[-1], 're_test'] = trend_row['re_test'] + 1
                # print(trends.iloc[-1]['re_test'])
            if trend_row['re_test'] >= re_min_tests:
                df.loc[bx:,'r1_trend'] = df.loc[bx:,trend_name]

            # check previous one by one in order of addition if they are our next/current support
            for it, t in trends[trends['su_test'] >= su_min_tests][::-1].iterrows():
                ax = int(t['ax'])
                ay = t['ay']
                if ny >= (df.loc[nx][t['name']] * 1-fake):
                    # print (t['name'])
                    df.loc[nx:,'s1_trend'] = df.loc[nx:,t['name']]
                    found_su = True
            # plot_trends(df, interval=interval, pair=pair, abn=[ax, bx, nx], i=i, filename='temp_plot'+str(i)+'.png')
            # if not found_su:
            #     ax = bx
            #     ay = by
            bx = nx
            by = ny
        nx = p.index[i]
        ny = p.iloc[i].low
        if i == last:
            b = False
        i += 1
    # print("--- %s seconds ---" % (time.time() - start_time))filename = 'chart_plots/' + pair.replace('/', '-') + '-' +  interval + \

    # Plot animation
    # filename = pair.replace('/', '-') + '-' +  interval + str(len(df)) + datetime.utcnow().strftime('-%H') + '.gif'
    # os.chdir('/home/bruno/Documentos/work/bitcoins/traders/freqtrade/chart_plots/')
    # subprocess.call(['ffmpeg', '-y', '-i', 'temp_plot%d.png', 'output.avi'])
    # subprocess.call(['ffmpeg', '-y', '-i', 'output.avi', '-t', '5', str(filename + '.gif')])
    # subprocess.call(['convert', '-delay 100', '*.png', filename])
    return df
