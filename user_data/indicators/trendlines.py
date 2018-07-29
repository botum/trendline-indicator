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

#     p = df.loc[df['pivots']==-1]

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


    df = get_fractals(df)


    r = df.loc[df[pivot_type]==1]
    rx = r.index[0]
    ry = r.iloc[0].high
    df.loc[0:, 'r1_trend'] = ry

    s = df.loc[df[pivot_type]==-1]

    # just working with supports now.
    p = s
    last=len(p)-3
    i=0

    prev_ai = []
    ai = i
    ax = p.index[i]
    ay = p.iloc[i].low
    df.loc[0:,'s1_trend'] = ay

    b=True
    bx = p.index[i+1]
    by = p.iloc[i+1].low

    while b:

        # trace first trend

        t = df.index[ax:]
        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend = polyval([slope,intercept],t)
        angle = calculate_angle(ax, ay, bx, by)
        trend_name = 'trend|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)+'|'+str(round(angle))
        df.loc[ax:,trend_name] = trend
        df.loc[ax:,'s1_trend'] = trend


        df = get_line_tests(df, trend_name, pivot_type, nearby, fake)
        data_name = 'data_'+ trend_name

        re_name = data_name+'|re'
        su_name = data_name+'|su'
        body_name = data_name+'|body'

        cx = p.index[i+2]
        cy = p.iloc[i+2].low

        cond = angle >= angle_min and angle <= angle_max \
                and np.count_nonzero(df[su_name]) >= su_min_tests

        if cy <= (df.loc[cx][trend_name] * (1-fake)):
            # we can use this support broken as resistance but better do it after a confirmation
            if cond == True and (df.iloc[cx][trend_name] * (1-fake)) <= df.iloc[cx]['r1_trend']:
                df.loc[cx:,'r1_trend'] = df.loc[cx:,trend_name]
            if len(prev_ai)>0:
                ai = prev_ai.pop()
                ax = p.index[ai]
                ay = p.iloc[ai].low
            bx = cx
            by = cy
        elif cy > (df.loc[cx][trend_name] * (1+nearby)):
            if cond == True and (df.iloc[cx][trend_name] * (1+nearby)) >= df.iloc[cx]['s1_trend']:
                df.loc[bx:,'s1_trend'] = df.loc[cx:,trend_name]
            prev_ai.append(ai)
            ax = bx
            ay = by
            ai = i
            bx = cx
            by = cy
        else:
            bx = cx
            by = cy

        if i == last:
            b = False
        i += 1

    print("--- %s seconds ---" % (time.time() - start_time))
    return df
