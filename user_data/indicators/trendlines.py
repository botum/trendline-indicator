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




def get_trends_OHCL(df: DataFrame, interval: int, tolerance: int, pivot_type: str, \
                     su_min_tests: int, re_min_tests: int, body_min_tests: int, \
                     ticker_gap: int, fake: int, nearby: int, angle_min: int, \
                     angle_max: int, thresh_up: int, thresh_down: int, \
                     chart=False, pair: str=None):
    ''' this approach is based on concurrence of points, either low high or body of candle'''

    start_time = time.time()
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

        for ib in range(0, len(next_waves)):
            # print (ib)
            bx = next_waves.index[ib]
            by = next_waves.iloc[ib].low

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

    print("--- %s seconds ---" % (time.time() - start_time))
    return df

def get_trends_lightbuoy_OHCL(df: DataFrame, interval: int, tolerance: int, pivot_type: str, \
                     su_min_tests: int, re_min_tests: int, body_min_tests: int, \
                     ticker_gap: int, fake: int, nearby: int, angle_min: int, \
                     angle_max: int, thresh_up: int, thresh_down: int, \
                     chart=False, pair: str=None):

    ''' this approach focus on the lower/higher trend progressivelly. Currently plotting after 2nd point
    as the third is our validation and we might want to buy or sell if broken.

    '''

    start_time = time.time()
    print('finding trends - lightbuoy style')

    if pivot_type == 'pivots':
        df, pivots = get_pivots_OHCL(df, period=20, stdv= 1.2,
                    interval=interval, thresh_type = 'dynamic',
                    thresh_up = 0.03, thresh_down = -0.03,
                    chart=False, pair=pair)
    elif pivot_type == 'fractals':
        df = get_fractals(df)

    # df = get_fractals(df)

    trends = []

    r = df.loc[df[pivot_type]==1]
    rx = r.index[0]
    ry = r.iloc[0].high
    df.loc[0:r.index[1], 'r1_trend'] = ry

    s = df.loc[df[pivot_type]==-1]

    # just working with supports now
    p = s
    last=len(p)-3
    i=0
    ai = i
    df.loc[0:p.index[1],'s1_trend'] = p.iloc[i].low

    prev_ai = []

    b=True

    while b:
        ax = p.index[i]
        ay = p.iloc[i].low

        bx = p.index[i+1]
        by = p.iloc[i+1].low

        # trace first trend
        t = df.index[ax:]
        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend = polyval([slope,intercept],t)
        angle = calculate_angle(ax, ay, bx, by)
        trend_name = 'trend|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)+'|'+str(round(angle))
        df.loc[ax:,trend_name] = trend

        df = get_line_tests(df, trend_name, pivot_type, nearby, fake)
        data_name = 'data_'+ trend_name

        re_name = data_name+'|re'
        su_name = data_name+'|su'
        body_name = data_name+'|body'

        trend = {'name':trend_name,
                'interval':interval,
                'a':[ax, ay, df.iloc[ax].date],
                'b':[bx, by, df.iloc[bx].date],
                'linregress'
                'angle':angle,
                'su_conf':np.count_nonzero(df[su_name]), # just so we know last totals
                're_conf':np.count_nonzero(df[re_name]),
                'body_conf':np.count_nonzero(df[body_name]),
                'type':'res',
                'last':False,
                'max':False,
                'min':False}
        trends.append(trend)

        # check indicator parameters
        cond = angle >= angle_min and angle <= angle_max \
                and np.count_nonzero(df[su_name]) >= su_min_tests

        # set it as support if valid
        if cond == True and (df.iloc[bx][trend_name]) <= df.iloc[bx]['low']:
            df.loc[bx:,'s1_trend'] = df.loc[bx:,trend_name]

        # next point
        nx = p.index[i+2]
        ny = p.iloc[i+2].low

        if ny <= (df.loc[nx][trend_name]):
            # we can use this support broken as resistance but better do it after a confirmation
            # still need to draw the line from the confirmation point, right now from third point nx
            if cond == True and (df.iloc[nx][trend_name]) >= df.iloc[nx]['high'] and angle < 90:
                df.loc[bx:,'r1_trend'] = df.loc[bx:,trend_name]
            if len(prev_ai)>0:
                ai = prev_ai.pop()
                ax = p.index[ai]
                ay = p.iloc[ai].low
            else:
                ai = i
                ax = p.index[ai]
                ay = p.iloc[ai].low
            bx = nx
            by = ny
        elif ny > (df.loc[nx][trend_name]):
            if cond == True and (df.iloc[nx][trend_name]) <= df.iloc[nx]['low'] and angle > 90:
                df.loc[nx:,'s1_trend'] = df.loc[nx:,trend_name]
            ax = bx
            ay = by
            prev_ai.append(i)
            bx = nx
            by = ny
        else:
            bx = nx
            by = ny

        if i == last:
            b = False
        i += 1
    print("--- %s seconds ---" % (time.time() - start_time))
    return df
