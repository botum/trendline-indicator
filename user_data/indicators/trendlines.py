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

import numpy as np
import pandas as pd

import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import arrow
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine, inspect)
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.pool import StaticPool

base = declarative_base()

def init() -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :return: None
    """
    db_url = "sqlite:///trends.sqlite"
    kwargs = {}

    # Take care of thread ownership if in-memory db
    if db_url == 'sqlite://':
        kwargs.update({
            'connect_args': {'check_same_thread': False},
            'poolclass': StaticPool,
            'echo': False,
        })

    try:
        engine = create_engine(db_url, **kwargs)
    except NoSuchModuleError:
        raise OperationalException(f'Given value for db_url: \'{db_url}\' '
                                   f'is no valid database URL! (See {_SQL_DOCS_URL})')

    session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trend.session = session()
    Trend.query = session.query_property()
    base.metadata.create_all(engine)


def cleanup() -> None:
    """
    Flushes all pending operations to disk.
    :return: None
    """
    Trend.session.flush()

class Trend(base):
    """
    Class used to define a trend. Any line we find a pattern starting from two
    (or min tests).
    """
    __tablename__ = 'trends'

    id = Column(Integer, primary_key=True)
    # index = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    pair = Column(String, nullable=False)
    serie = Column(String, nullable=False)
    parent = Column(Integer, nullable=True)
    ad = Column(DateTime, nullable=False)
    ay = Column(Float, nullable=True, default=0.0)
    bd = Column(DateTime, nullable=False)
    by = Column(Float, nullable=True, default=0.0)
    angle = Column(Float, nullable=True, default=0.0)
    su_tests = Column(Integer, primary_key=True)
    re_tests = Column(Integer, primary_key=True)
    body_tests = Column(Integer, primary_key=True)
    volume = Column(Float, nullable=True, default=0.0)
    interval = Column(Integer, nullable=True)


    def __repr__(self):
        return (f'Trend(id={self.id}, pair={self.pair}, ad={self.ad}, '
                f'bd={self.bd}, )')

    def update(self, df):
        """Find new waves and create new lightbuoys"""

        # get last lightbuoy
        return None

    def get_tests_volume(self, rate: float) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        return None

    def get_parents(self, rate: float) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        return None

def plot_trends(df, interval: int, abn: int=0, pair: str=None, filename: str=None, cols=[], i: int=0, debug: bool=False):
    plt.figure(num=0, figsize=(20,10))
    # df['old_date'] = df['date']
    # pd.to_datetime(df['date'])
    # df.set_index(['date'],inplace=True)
#     for c in cols:
#         plt.scatter(df.index, df[c], color='r', label=c, s=10)


    if 's1_trend' in df:
        plt.plot(df['s1_trend'], color='g', label='su trend', linewidth=1)
    if 'r1_trend' in df:
        plt.plot(df['r1_trend'], color='r', label='re trend', linewidth=1)
    if 's1_trend_conf' in df:
        plt.plot(df['s1_trend_conf'], color='g', label='su conf trend', linewidth=2)
    if 's1_trend_not_conf' in df:
        plt.plot(df['s1_trend_not_conf'], color='b', label='su not conf trend', linewidth=2)
    if 'r1_trend_conf' in df:
        plt.plot(df['r1_trend_conf'], color='r', label='re conf trend', linewidth=2)
    if 'r1_trend_not_conf' in df:
        plt.plot(df['r1_trend_not_conf'], color='b', label='re not conf trend', linewidth=2)
    plt.plot(df.high, 'r', alpha=0.5)
    plt.plot(df.close, 'k', alpha=0.5)
    plt.plot(df.low, 'g', alpha=0.5)

#     plt.plot(df.bb_lower, 'b', alpha=0.5, linewidth=2)
#     plt.plot(df.bb_upper, 'b', alpha=0.5, linewidth=2)

    trends = [col for col in df if col.startswith('trend_')]
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

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if debug:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()

def trend_to_dataframe(serie: pd.Series, a: list, b: list):
    slope, intercept, r_value, p_value, std_err = stats.linregress([a[0], b[0]], [a[1], b[1]])
    trend = pd.Series(polyval([slope,intercept],serie))
    # print('len df: ',len(serie))
    return trend

def point_in_trend(a, b, n):
    slope, intercept, r_value, p_value, std_err = stats.linregress([a[0], b[0]], [a[1], b[1]])
    return polyval([slope,intercept],[n[0]])

def is_above(p,a,b):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    p = np.array([p[0], p[1]])
    return np.cross(p-a, b-a) <= 0


from abc import ABCMeta, abstractmethod

class Pivot(object):
    """A point of change of direction in market.


    Attributes:
        pair: .
        x: Date.
        y: Price.
        direction: Support or resistance.
        volume: Volume traded in that tick.
    """

    __metaclass__ = ABCMeta

    x = 0
    y = 0
    direction = 0
    volume = 0

    def __init__(self, pair, x, y, direction, volume):
        self.pair = pair
        self.x = x
        self.y = y
        self.direction = direction
        self.volume = volume

    def x_in_dataframe(self, df):
        """Return the x position of this date in the incomming DF."""

        return df

    # def purchase_price(self):
    #     """Return the price for which we would pay to purchase the vehicle."""
    #     if self.sold_on is None:
    #         return 0.0  # Not yet sold
    #     return self.base_sale_price - (.10 * self.miles)
    #
    # @abstractmethod
    # def vehicle_type(self):
    #     """"Return a string representing the type of vehicle this is."""
    #     pass

class Pivot():
    def __init__(self):
        self.__pList = []
    def addPerson(self,name,number):
        self.__pList.append(Person(name,number))
    def findPerson(self, **kwargs):
        return next(self.__iterPerson(**kwargs))
    def allPersons(self, **kwargs):
        return list(self.__iterPerson(**kwargs))
    def __iterPerson(self, **kwargs):
        return (person for person in self.__pList if person.match(**kwargs))

class Person():
    def __init__(self,name,number):
        self.nom = name
        self.num = number
    def __repr__(self):
        return "Person('%s', %d)" % (self.nom, self.num)
    def match(self, **kwargs):
        return all(getattr(self, key) == val for (key, val) in kwargs.items())
class Trend(object):
    """A point of change of direction in market.


    Attributes:
        pair: .
        x: Date.
        y: Price.
        direction: Support or resistance.
        volume: Volume traded in that tick.
    """

    __metaclass__ = ABCMeta

    x = 0
    y = 0
    direction = 0
    volume = 0

    def __init__(self, pair, x, y, direction, volume):
        self.pair = pair
        self.x = x
        self.y = y
        self.direction = direction
        self.volume = volume

    def x_in_dataframe(self, df):
        """Return the x position of this date in the incomming DF."""

        return df
class Car(Vehicle):
    """A car for sale by Jeffco Car Dealership."""

    base_sale_price = 8000
    wheels = 4

    def vehicle_type(self):
        """"Return a string representing the type of vehicle this is."""
        return 'car'

def get_trends_lightbuoy_OHCL(df: pd.DataFrame, interval: int, pressision: int, pivot_type: str, \
                     su_min_tests: int, re_min_tests: int, body_min_tests: int, \
                     ticker_gap: int, fake: int, nearby: int, angle_min: int, \
                     angle_max: int, thresh_up: int, thresh_down: int, \
                     chart=False, plot_animation: bool=False, pair: str=None, \
                     debug: bool=False ):

    ''' this approach focus on the lower/higher trend progressivelly. Currently plotting after 2nd point
    as the third is our validation and we might want to buy or sell if broken.

    '''

    # start_time = time.time()
    # print('finding trends - lightbuoy style')

    # trends = pd.read_sql(Trend.session.query(Trend).filter(Trend.pair == pair).statement,Trend.session.bind)
    os.chdir('/home/bruno/Documentos/work/bitcoins/traders/freqtrade/chart_plots/')


    pivot_filename = pair.replace('/', '-') + pivot_type + '.json'
    print ('try loading pivots ', pivot_filename)

    try:
        with open(pivot_filename, 'r') as f:
            data = json.load(f)
            df = df.append(data)
    except Exception as e:
        print ('exception: ', e)
        if pivot_type == 'pivots':
            df, pivots = get_pivots_OHCL(df, period=20, stdv= .8,
                        interval=interval, thresh_type = '',
                        thresh_up = thresh_up, thresh_down = thresh_down,
                        chart=False, pair=pair)
        elif pivot_type == 'fractals':
            df = get_fractals(df)

    s = df.loc[df[pivot_type]==-1]

    # just working with supports now
    p = s
    # df = get_fractals(df)

    trend_filename = pair.replace('/', '-') + '_trends.json'
    print ('try loading trends ', trend_filename)

    try:
        with open(trend_filename, 'r') as f:
            trends = pd.read_json(f, orient='records', lines=True)
            trends.ad = pd.to_datetime(trends.ad.tz_localize(None))
            trends.bd = pd.to_datetime(trends.bd.tz_localize(None))
    except Exception as e:
        print ('exception: ', e)
        trends = pd.DataFrame({})

    if len(trends.index) > 0:
        print (len(trends.index), ' trends loaded')
        current_trend = trends.iloc[-1]
        ad = current_trend['ad']
        ax = df.index[df['date'] == current_trend['ad']]
        ay = current_trend['ay']
        bd = current_trend['bd']
        bx = df.index[df['date'] == current_trend['bd']]
        by = current_trend['by']
        nx = p.index[2]
        ny = p.iloc[2].low
        i = len(p) - p.index[bx]
        last=len(p)
    else:
        # print ('current trends: ',trends)
        r = df.loc[df[pivot_type]==1]
        rx = r.index[0]
        ry = r.iloc[0].high

        ax = p.index[0]
        ad = p.loc[ax].date
        ay = p.iloc[0].low
        new_lightbuoy = True # we suppose we start on a lightbuoy

        bx = p.index[1]
        bd = p.loc[bx].date
        by = p.iloc[1].low

        # right now to get point N we have to wait for the next height pivot point,
        # this could be done with fractals
        nx = p.index[2]
        ny = p.iloc[2].low

        last=len(p)
        i = 2

        # initial sure
        df.loc[0:r.index[1], 'r1_trend'] = ry
        df.loc[0:p.index[1],'s1_trend'] = p.iloc[0].low


    a = [ax,ay]
    b = [bx,by]
    n = [nx,ny]
        # plot_trends(df, interval=interval, pair=pair, abn=nx)

    next_pivot=True
    while next_pivot:
        # print ('start ---', i)
        # trace first trend
        if plot_animation:
            plot_trends(df, interval=interval, pair=pair, abn=[a[0], b[0], n[0]], i=i, filename='temp_plot'+str(i)+'.png')
        if i == last:
            nx = df.index[-1]
            ny = df.iloc[-1].low
        else:
            nx = p.index[i]
            ny = p.iloc[i].low

        n = [nx,ny]
        # df, trend_name, angle = trend_to_dataframe(df, a, b)
        # print('ax: ',ax)
        # print('bx: ',bx)
        # print(p)

        angle = calculate_angle(a, b)

        trend_name = 'trend_'+pivot_type+'_'+ad.strftime("%s")+'_'+bd.strftime("%s")

        ad = p.loc[a[0]].date
        bd = p.loc[b[0]].date
        trends = trends.append({
                'name':trend_name,
                'interval':interval[:-1],
                'ad':ad,
                'ay':ay,
                'bd':bd,
                'by':by,
                'angle':angle,
                'su_tests':[a, b], # we already have two touches.
                're_tests':[],
                'body_tests':0
                }, ignore_index=True)
        trend_row = trends.iloc[-1]

        # print('len df.loc: ',len(df.loc[a[0]:]))
        df.loc[a[0]:,trend_name] = trend_to_dataframe(df.index, a, b)

        if debug:
            plot_trends(df, interval=interval, pair=pair, abn=[a[0], b[0], n[0]],
            i=i, filename='temp_plot'+str(i)+'.png', debug=debug)
            print ('ax: ', ax)
            print ('ay: ', ay)
            print ('bx: ', bx)
            print ('by: ', by)
            print ('nx: ', nx)
            print ('ny: ', ny)
        cond = angle >= angle_min and angle <= angle_max

        ty = point_in_trend(a,b,n)

        if is_above(n,a,b) and cond:
            # print('going up')
            if in_range(ny, ty * 1-fake, pressision):
                conf_list = trend_row['su_tests']
                conf_list.append(n)
                trends.at[int(trends.index[-1]), 'su_tests'] = conf_list
                su_tests = len(trend_row['su_tests'])
            # prev_ai.append(p.index[i+1])
                # else:
                # cond =  angle >= angle_min and angle <= angle_max \
                #         and trend['su_test'] >= su_min_tests
                #         # and (df.iloc[nx][trend_name]) >= df.iloc[nx]['low']
            if su_tests >= su_min_tests and cond:
                conf_point = trend_row['su_tests'][su_min_tests-1]
                df.loc[conf_point[0]:n[0],'s1_trend_conf'] = trend_to_dataframe(df.index, a, b)
                # print(conf_point)
            else:
                conf_point = trend_row['su_tests'][-1]
                df.loc[conf_point[0]:n[0],'s1_trend_not_conf'] = trend_to_dataframe(df.index, a, b)
                df.loc[conf_point[0]:n[0],'s1_trend'] = trend_to_dataframe(df.index, a, b)

            # df.loc[ax:,'s1_trend'] = df.loc[ax:,trend_name]
            if debug:
                plot_trends(df, interval=interval, pair=pair, abn=[a[0], b[0], n[0]],
                i=i, filename='temp_plot'+str(i)+'.png', debug=debug)
                print ('ax: ', ax)
                print ('ay: ', ay)
                print ('bx: ', bx)
                print ('by: ', by)
                print ('nx: ', nx)
                print ('ny: ', ny)

            a = b
            b = n

            if new_lightbuoy:
                new_lightbuoy = False

            # print (trend_row)
        else:
            # print('going down')
            found_su = False
            # df.loc[nx:,'s1_trend'] = np.nan
            # if cond == True and (df.iloc[nx][trend_name]) <= df.iloc[nx]['low'] and angle > 90:
            #     df.loc[bx:,'s1_trend'] = df.loc[bx:,trend_name]
            if in_range(df.loc[nx]['high'], ty, pressision):
                # print(trend_row)
                trends.at[trends.index[-1], 're_tests'] = trend_row['re_tests'].append(n)
                # print(trend_row)
                re_tests = int(len(trend_row['re_tests']))
                # print(trends.iloc[-1]['re_test'])
            if re_tests >= re_min_tests and cond:
                df.loc[n[0]:, 'r1_trend_conf'] = trend_to_dataframe(df.index, a, b)
            else:
                df.loc[n[0]:, 'r1_trend_not_conf'] = trend_to_dataframe(df.index, a, b)
            # check previous one by one in order of addition if they are our next/current support
            for it, t in trends.iloc[::-1].iterrows():
                # print(df[df['date']==t['ad']].index.values.astype(int)[0])
                st_ax = df[df['date']==t['ad']].index.values.astype(int)[0]
                st_ay = t['ay']
                st_a = [st_ax,st_ay]
                # print ('test support st_ax: ', st_ax)
                st_bx = df[df['date']==t['bd']].index.values.astype(int)[0]
                st_by = t['by']
                st_b = [st_bx,st_by]
                su_trend_name = 'trend_'+pivot_type+'_'+ad.strftime("%s")+'_'+bd.strftime("%s")
                df.loc[st_a[0]:, su_trend_name] = trend_to_dataframe(df.index, st_a, st_b)
                if debug:
                    plot_trends(df, interval=interval, pair=pair, abn=[st_a[0], st_b[0], n[0]],
                    i=i, filename='temp_plot'+str(i)+'.png', debug=debug)
                    print ('st_ax: ', st_ax)
                    print ('st_ay: ', st_ay)
                    print ('st_bx: ', st_bx)
                    print ('st_by: ', st_by)
                    print ('nx: ', nx)
                    print ('ny: ', ny)
                if is_above(n,st_a,st_b) and cond:
                    # print ('support found: ', t['name'])
                    if len(t['su_tests']) >= su_min_tests:
                        conf_point = t['su_tests'][su_min_tests-1]
                        df.loc[n[0]:, 's1_trend_conf'] = trend_to_dataframe(df.index, st_a, st_b)
                        a = st_a
                        b = st_b
                        found_su = True
                        break
                    else:
                        conf_point = t['su_tests'][-1]
                        df.loc[conf_point[0]:, 's1_trend_not_conf'] = trend_to_dataframe(df.index, st_a, st_b)
                        df.loc[conf_point[0]:, 's1_trend'] = trend_to_dataframe(df.index, st_a, st_b)
                    # new_lightbuoy = True
            # check back A points.
            if not found_su:
                # print ('no support, search for min trend')
                # past_lb = trends.drop_duplicates('ad')
                # print ('past: ',past_lb)
                lb_n_angles = np.array([])
                for ilb, lightbuoy in trends.iloc[::-1].iterrows():
                    lb_ax = df[df['date']==lightbuoy['ad']].index.values.astype(int)[0]
                    lb_ay = lightbuoy['ay']
                    lb_a = [lb_ax,lb_ay]

                    angle = calculate_angle(a, n)
                    min_su_trend_name = 'trend_'+pivot_type+'_'+ad.strftime("%s")+'_'+bd.strftime("%s")
                    lb_n_angles = np.append(lb_n_angles, [lightbuoy.name, angle], axis=0)
                    df.loc[lb_a[0]:, min_su_trend_name] = trend_to_dataframe(df.index, lb_a, st_b)
                    if debug:
                        plot_trends(df, interval=interval, pair=pair, abn=[lb_a[0], b[0], n[0]],
                        i=i, filename='temp_plot'+str(i)+'.png', debug=debug)
                        print ('lb_ax: ', lb_ax)
                        print ('lb_ay: ', lb_ay)
                        print ('bx: ', bx)
                        print ('by: ', by)
                        print ('nx: ', nx)
                        print ('ny: ', ny)
                    # future_nx = nx + 10
                    # lb_ny = point_in_trend(lb_a,lb_b,future_n)
                    #
                    # if ny >= (df.loc[nx][sup_name] * 1-fake) and df.loc[nx][trend_name] >= df.loc[nx]['s1_trend']:
                    # # if ax != int(s['ax']):
                    #     df.loc[bx:,'s1_trend'] = df.loc[bx:,sup_name]
                    #     found_su = True
                    #     # print (ax)
                    #     # new_lightbuoy = True
                    # ax = lbx
                    # ay = lby
                min_trendline = trends.loc[int(np.amin(lb_n_angles, axis=0))]
                ax = df[df['date']==min_trendline['ad']].index.values.astype(int)[0]
                ay = min_trendline['ay']
                a = [ax, ay]
                b = n
                df.loc[a[0]:, 'min_trend'] = trend_to_dataframe(df.index, a, b)

            # if not found_su:
            #     print ('last support')
            #
            #     st_ax = df[df['date']==t['ad']].index.values.astype(int)[0]
            #     df.loc[b[0]:] = trend_to_dataframe(df.loc[b[0]:], 's1_trend_not_conf', a, n)
            #     # df.loc[bx:,'s1_trend'] = df.loc[bx:,trend_name]
            #     new_lightbuoy = True
            #     # ax = bx
            #     # ay = by
            if debug:
                plot_trends(df, interval=interval, pair=pair, abn=[a[0], b[0], n[0]],
                            i=i, filename='temp_plot'+str(i)+'.png', debug=debug)
                print ('ax: ', ax)
                print ('ay: ', ay)
                print ('bx: ', bx)
                print ('by: ', by)
                print ('nx: ', nx)
                print ('ny: ', ny)
            # if not found_su:
            #     ax = bx
            #     ay = by
        # bx = nx
        # by = ny
        # print ('end ---', i)
        if i == last:
            next_pivot = False
        else:
            i += 1
    # df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    # trends.to_sql('trends', con=Trend.session.bind, if_exists='replace', index_label='id')
    # print("--- %s seconds ---" % (time.time() - start_time))filename = 'chart_plots/' + pair.replace('/', '-') + '-' +  interval + \
    with open(trend_filename, 'w') as f:
        f.write(trends.to_json(orient='records', lines=True, date_unit='ns'))
    with open(pivot_filename, 'w') as f:
        f.write(df[pivot_type].to_json(orient='records', lines=True, date_unit='ns'))

    # Plot animation
    if plot_animation:
        filename = pair.replace('/', '-') + '-' +  interval + str(len(df)) + '.avi'
        subprocess.call(['ffmpeg', '-y', '-r', '1', '-i', 'temp_plot%d.png', filename])
        subprocess.call('rm -rf *.png', shell=True)
        # convert to gif
        # subprocess.call(['ffmpeg', '-y', '-i', 'output.avi', '-t', '5', str(filename + '.gif')])
        # doit with magic
        # subprocess.call(['convert', '-delay 100', '*.png', filename])
    return df


init()
