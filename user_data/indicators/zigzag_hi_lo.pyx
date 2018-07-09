cimport cython
import numpy as np
from numpy cimport ndarray, int_t
from freqtrade.vendor.qtpylib.indicators import numpy_rolling_mean

# customized zig-zag lib for using high/low and dynamic threshold

DEF PEAK = 1
DEF VALLEY = -1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int_t identify_initial_pivot(double [:] L,
                                   double [:] H,
                                   double up_thresh,
                                   double down_thresh):
    cdef:
        double x_0 = L[0]
        double x_t = x_0

        double max_x = x_0
        double min_x = x_0

        int_t max_t = 0
        int_t min_t = 0

    up_thresh += 1
    down_thresh += 1
    # print (up_thresh)
    # print (down_thresh)

    for t in range(1, len(L)):
        l_t = L[t]
        h_t = H[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = h_t
            max_t = t

        if x_t < min_x:
            min_x = l_t
            min_t = t

    t_n = len(L)-1
    return VALLEY if x_0 < L[t_n] else PEAK


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef peak_valley_pivots(double [:] L,
                         double [:] H,
                         double [:] E):
    """
    Find the peaks and valleys of a series.

    :param X: the series to analyze
    :param up_thresh: minimum relative change necessary to define a peak
    :param down_thesh: minimum relative change necessary to define a valley
    :return: an array with 0 indicating no pivot and -1 and 1 indicating
        valley and peak


    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias
    analysis.
    """
#     if down_thresh > 0:
#         raise ValueError('The down_thresh must be negative.')

    cdef:
        double up_thresh = E[0]
        double down_thresh = E[0]
        int_t initial_pivot = identify_initial_pivot(L, H,
                                                     up_thresh,
                                                     down_thresh)
        int_t t_n = len(L)
        ndarray[int_t, ndim=1] pivots = np.zeros(t_n, dtype=np.int_)
        int_t trend = -initial_pivot
        int_t last_pivot_t = 0
        double last_pivot_x = L[0]
        double x, r

    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.

    for t in range(1, t_n):
        x = L[t]
        r = x / last_pivot_x

        up_thresh = E[t] + 1
        down_thresh = -E[t] + 1

        # print ('upd: ', up_thresh)
        # print ('downd: ', down_thresh)

        if trend == -1:
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = PEAK
                last_pivot_x = x
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = VALLEY
                last_pivot_x = x
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t

    # if last_pivot_t == t_n-1:
    #     pivots[last_pivot_t] = trend
    # elif pivots[t_n-1] == 0:
    #     pivots[t_n-1] = -trend

    return pivots
