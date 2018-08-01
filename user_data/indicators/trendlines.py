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
    '''structure:
    - confirmed_trends[{'ax': ax, 'ay':ay...}, {}, {}]
    '''
    trends = DataFrame(columns=[
            'name',
            'interval',
            'a',
            'b',
            'angle',
            'su_conf',
            're_conf',
            'body_conf'
            ])

    r = df.loc[df[pivot_type]==1]
    rx = r.index[0]
    ry = r.iloc[0].high
    df.loc[0:r.index[1], 'r1_trend'] = ry

    s = df.loc[df[pivot_type]==-1]

    # just working with supports now
    p = s
    last=len(p)-2
    ia = 0
    ib = 0
    df.loc[0:p.index[1],'s1_trend'] = p.iloc[0].low

    ax = p.index[0]
    ay = p.iloc[0].low

    bx = p.index[1]
    by = p.iloc[1].low

    b=True

    while b:

        # trace first trend
        t = df.index[ax:]
        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend = polyval([slope,intercept],t)
        angle = calculate_angle(ax, ay, bx, by)
        trend_name = 'trend|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)+'|'+str(round(angle))
        df.loc[ax:,trend_name] = trend

        # df = get_line_tests(df, trend_name, pivot_type, nearby, fake)
        data_name = 'data_'+ trend_name

        re_name = data_name+'|re'
        su_name = data_name+'|su'
        body_name = data_name+'|body'

        trends = trends.append({
                'name':trend_name,
                'interval':interval,
                'a':[ax, ay],
                'b':[bx, by],
                'angle':angle,
                'su_test':2, # just so we know last totals
                're_test':0,
                'body_test':0
                }, ignore_index=True)

        trend_row = trends.iloc[-1]
        # next point
        nx = p.index[ib+1]
        ny = p.iloc[ib+1].low
        if ny >= (df.loc[nx][trend_name]):
            if in_range(ny, df.loc[nx][trend_name], pressision):
                tests = trends.at[trend_row.index[0], su_name] = trends.iloc[-1]['su_test'] + 1
            # else:
                # cond =  angle >= angle_min and angle <= angle_max \
                #         and trend['su_test'] >= su_min_tests
                #         # and (df.iloc[nx][trend_name]) >= df.iloc[nx]['low']
                # if tests >= su_min_tests :
                    # df.loc[bx:,'r1_trend'] = df.loc[bx:,trend_name]
            df.loc[nx:,'s1_trend'] = df.loc[nx:,trend_name]
            ax = bx
            ay = by
            bx = nx
            by = ny
        else:
            # if cond == True and (df.iloc[nx][trend_name]) <= df.iloc[nx]['low'] and angle > 90:
            #     df.loc[nx:,'s1_trend'] = df.loc[nx:,trend_name]
            if in_range(df.loc[nx]['high'], df.loc[nx][trend_name], pressision):
                tests = trends.at[trend_row.index[0], re_name] = trends.iloc[-1]['re_test'] + 1
            if tests >= re_min_tests :
                # df.loc[bx:,'r1_trend'] = df.loc[bx:,trend_name]
                df.loc[nx:,'r1_trend'] = df.loc[nx:,trend_name]
            prev_trends = [col for col in df if col.startswith('trend|')]
            # print (prev_trends)
            # print('prev_trend: ',prev_trends.ix[nx,prev_trends.lt(ny).max()])
            # bx = nx
            # by = ny
            # else:
            #     bx = nx
            #     by = ny
            # ax = bx
            # ay = by
            # prev_ai.append(i)
            # bx = nx
            # by = ny
            bx = nx
            by = ny
        if ib == last:
            # print (trends)
            b = False
        ib += 1
    print("--- %s seconds ---" % (time.time() - start_time))
    return df
