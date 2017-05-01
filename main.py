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

######
# MPI
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

#####
from evaluation import *
from basic_processing import *
from bayesian_regression import *

comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

def main():

    if comm.rank == 0:
        T1 = MPI.Wtime();
    
    ######
    # TODO general IO, with [nstock, ntslen]
    if comm.rank == 0:

        # one_stock is a np.array([])
        one_stock = io_price("AAPL")

        # price_init is in float64 type, price_init[1:npts]
        price_init = collapse(one_stock, time_intval)[offset:offset+npts]

        ######
        # shift by 500 to skip the big dip at the begining of source data
        # ??????
        # Why shift?
        ######
        two_stock = io_price("GOOG")
        price2_init = collapse(two_stock, time_intval)[offset-ts3len:offset-ts3len+npts]

        # Divide prices into two, roughly equal sized, periods:
        
        price = price_init[:npts-nval];
        price2 = price2_init[:npts-nval];

        ######
        # TODO
        ######
        # price_global = price_init[1:nstock_global,1:ntime]
        # price_proc = price_init[1:nstock_proc,1:ntime]
        
        price_val = price_init[npts-nval-ts3len:];
        #price_val = price_init

        ######
        # [Debug]
        # print out data type
        # print(price.dtype) => float 64
        # print(price_val.dtype) => float 64
        # print("len[price_init] = ", len(price_init))
        # print("len[price_val] = ", len(price_val))        
        ######        
    else :
        price = np.empty(npts-nval,dtype='float64');
        price2 = np.empty(npts-nval,dtype='float64');
        price_val = np.empty(nval+ts3len,dtype='float64');
    
    ######
    # Broadcast price, price2, price_val
    comm.Bcast(price,root=0)
    comm.Bcast(price2,root=0)
    comm.Bcast(price_val,root=0)

    if comm.rank == 0:
        T2 = MPI.Wtime()
        print("Time[ Read in stock data ] : ", T2 - T1)
    
    # print(price[233]," proc = ", comm.rank)
    # print(price2[233]," proc = ", comm.rank)
    # print(price_val[233]," proc = ", comm.rank)

    ######
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

    print("Generation of timeseries done. proc = ",comm.rank)

    # ######
    # # find_cluster_centers will use KMeans, which involves a random number generator
    # # we set the random seed as : np.random.seed(23)
    # ######

    ######
    # TODO, how to parallize this part?
    # For AAPL
    # centers1 = find_cluster_centers(ts1[0], ts1[1], nkmean)
    # centers2 = find_cluster_centers(ts2[0], ts2[1], nkmean)
    # centers3 = find_cluster_centers(ts3[0], ts3[1], nkmean)

    # # For GOOG
    # centers4 = find_cluster_centers(ts4[0], ts4[1], nkmean)
    # centers5 = find_cluster_centers(ts5[0], ts5[1], nkmean)
    # centers6 = find_cluster_centers(ts6[0], ts6[1], nkmean)

    # print("%d cluster centers found." % nkmean, " proc = ", comm.rank)

    # ######
    # s_list = np.zeros((nstock*nts,ncenter));
    
    # s_list[0] = choose_effective_centers(centers1, ncenter, m_top_discard);
    # s_list[1] = choose_effective_centers(centers2, ncenter, m_top_discard);
    # s_list[2] = choose_effective_centers(centers3, ncenter, m_top_discard);

    # s_list[3] = choose_effective_centers(centers4, ncenter, m_top_discard);
    # s_list[4] = choose_effective_centers(centers5, ncenter, m_top_discard);
    # s_list[5] = choose_effective_centers(centers6, ncenter, m_top_discard);

    # print("%d cluster centers selected." % ncenter, " proc = ", comm.rank)
    # # print("s1, s2, s3", s1, s2, s3)

    # if comm.rank == 0:
    #     T3 = MPI.Wtime();
    #     print("Time[ Find centers ] : ", T3 - T2);

    # #exit(0)

    # ######
    # # Why here there is only price without price2 ???
    # ######
    # # [Parallelized]
    # ## Dpi_r, Dp = linear_regression_vars(price[len1/2:], s1, s2, s3, s4, s5, s6, time_window)
    # # print("Bayesian regression done.")

    # w = linear_regression_vars_find_w(price[len1/2:], s_list, time_window)

    # if comm.rank == 0:
    #     T4 = MPI.Wtime();
    #     print("Time[ Bayesian + linear regression ] : ", T4 - T3)

    # ## Debug
    # print("proc = ", comm.rank, " w = ", w);
        
    # ## Put find_w_linear_regression inside linear_regression_vars
    # ## w = find_w_linear_regression(Dpi_r, Dp)

    # ##    print("Linear regression done.", w)

    # ##    dps, pprice = predict_dps(price, s1, s2, s3, w, time_window)
    # # print("Done training. Start Validation.")

    # ######
    # # TODO, parallel
    # ######
    # #dps_val, pprice_val = predict_dps(price_val, s1, s2, s3, s4, s5, s6, w, time_window)

    # # dprice = diff_price(price_val)
    # # mdprice = mean_price(dprice, 20)

    # # #    plt.plot(dps_val)
    # # plt.plot(pprice_val)
    # # #    plt.plot(dprice)
    # # #plt.plot(mdprice)
    # # plt.plot(price_val)
    # # plt.show()

    # # plt.plot(dprice[ts3len+time_window:])
    # # plt.plot(dps_val)
    # # plt.show()

    # # # evaluation 1 -- original
    # # #    bank_balance = evaluate_performance(price, dps, ts3len, time_window, t=0.001, step=1)
    # # #    print("Final Balance:", bank_balance)

    # # # evaluation 2 -- track the change of total asset
    # # all_asset = evaluate_performance_asset(price_val, dps_val, ts3len, time_window, t=0.0001, step=time_window)
    # # plt.plot(all_asset/10.0)
    # # plt.plot(collapse(price_val[ts3len:], time_window))
    # # plt.title('Total asset')
    # # plt.show()
    # # print('Start Asset: %f' % all_asset[0])
    # # print('Final Asset: %f' % all_asset[-1])
    # # print("Weights", w)

    # comm.Barrier()

    # if comm.rank == 0:
    #     Ttotal = MPI.Wtime()
    #     print("Time[ Total ] : ", Ttotal - T1)


if __name__ == '__main__':
    time_intval = 1.0 # basic unit
    time_window = 5 # in time_intval unit
    npts = 10000  # including nval, and the rest: half to find centers, half to do linear regression
    nval = 2000

    nstock = 2;
    stockname=["AAPL","GOOG"];
    
    tslen = np.array([20, 60, 120])
    nslice = len(tslen);
    
    ts1len, ts2len, ts3len = 20, 60, 120;

    offset = ts3len + 20000
    nkmean, ncenter, m_top_discard = 100, 60, 2 # discard top m centers, because these are likely rare

    print("rank = ", comm.rank, "size = ", comm.size);
    print("Parameter list");
    print("npts = ", npts, "offset = ", offset);
    print("ts1len = ", ts1len, "ts2len = ", ts2len, "ts3len = ", ts3len)
    print("nkmean = ", nkmean, "ncenter = ", ncenter, "m_top_discard = ", m_top_discard);

    main()
