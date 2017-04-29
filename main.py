#!/usr/bin/python

# coding: utf-8
# Running the entire model

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cluster import KMeans
from skimage.feature import hog
import math
import pandas as pd

from evaluation import *
from basic_processing import *
from bayesian_regression import *

def main():
    one_stock = io_price("AAPL")
    price_init = collapse(one_stock, time_intval)[offset:offset+npts]
    # shift by 500 to skip the big dip at the begining of source data
    two_stock = io_price("GOOG")
    price2_init = collapse(two_stock, time_intval)[offset-ts3len:offset-ts3len+npts]

	# Divide prices into two, roughly equal sized, periods:

    price = price_init[:npts-nval]
    price2 = price2_init[:npts-nval]

    price_val = price_init[npts-nval-ts3len:]
    #price_val = price_init

    len1 = len(price)
    len2 = len(price2)

    # For AAPL
    ts1 = generate_timeseries(price[:len1/2], ts1len, time_window)
    ts2 = generate_timeseries(price[:len1/2], ts2len, time_window)
    ts3 = generate_timeseries(price[:len1/2], ts3len, time_window)
    # For GOOG
    ts4 = generate_timeseries(price2[:len2/2], ts1len, time_window)
    ts5 = generate_timeseries(price2[:len2/2], ts2len, time_window)
    ts6 = generate_timeseries(price2[:len2/2], ts3len, time_window)

    print("Generation of timeseries done.")

    # For AAPL
    centers1 = find_cluster_centers(ts1[0], ts1[1], nkmean)
    centers2 = find_cluster_centers(ts2[0], ts2[1], nkmean)
    centers3 = find_cluster_centers(ts3[0], ts3[1], nkmean)
    # For GOOG
    centers4 = find_cluster_centers(ts4[0], ts4[1], nkmean)
    centers5 = find_cluster_centers(ts5[0], ts5[1], nkmean)
    centers6 = find_cluster_centers(ts6[0], ts6[1], nkmean)

    print("%d cluster centers found." % nkmean)

    s1 = choose_effective_centers(centers1, ncenter, m_top_discard)
    s2 = choose_effective_centers(centers2, ncenter, m_top_discard)
    s3 = choose_effective_centers(centers3, ncenter, m_top_discard)

    s4 = choose_effective_centers(centers4, ncenter, m_top_discard)
    s5 = choose_effective_centers(centers5, ncenter, m_top_discard)
    s6 = choose_effective_centers(centers6, ncenter, m_top_discard)

    print("%d cluster centers selected." % ncenter)
#    print("s1, s2, s3", s1, s2, s3)

#    exit(0)


    Dpi_r, Dp = linear_regression_vars(price[len1/2:], s1, s2, s3, s4, s5, s6, time_window)
    print("Bayesian regression done.")

    w = find_w_linear_regression(Dpi_r, Dp)
#    print("Linear regression done.", w)

#    dps, pprice = predict_dps(price, s1, s2, s3, w, time_window)
    print("Done training. Start Validation.")
    dps_val, pprice_val = predict_dps(price_val, s1, s2, s3, s4, s5, s6, w, time_window)

    dprice = diff_price(price_val)
    mdprice = mean_price(dprice, 20)

#    plt.plot(dps_val)
    plt.plot(pprice_val)
#    plt.plot(dprice)
    #plt.plot(mdprice)
    plt.plot(price_val)
    plt.show()

    plt.plot(dprice[ts3len+time_window:])
    plt.plot(dps_val)
    plt.show()

    # evaluation 1 -- original
#    bank_balance = evaluate_performance(price, dps, ts3len, time_window, t=0.001, step=1)
#    print("Final Balance:", bank_balance)

    # evaluation 2 -- track the change of total asset
    all_asset = evaluate_performance_asset(price_val, dps_val, ts3len, time_window, t=0.0001, step=time_window)
    plt.plot(all_asset/10.0)
    plt.plot(collapse(price_val[ts3len:], time_window))
    plt.title('Total asset')
    plt.show()
    print('Start Asset: %f' % all_asset[0])
    print('Final Asset: %f' % all_asset[-1])
    print "Weights", w


if __name__ == '__main__':
    time_intval = 1.0 # basic unit
    time_window = 5 # in time_intval unit
    npts = 10000  # including nval, and the rest: half to find centers, half to do linear regression
    nval = 2000
    ts1len, ts2len, ts3len = 20, 60, 120
    offset = ts3len + 20000
    nkmean, ncenter, m_top_discard = 100, 60, 2 # discard top m centers, because these are likely rare
    main()
