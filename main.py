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
        # Start clock
        T1 = MPI.Wtime();

    if comm.rank == 0:
        # Prepare ss_start_global, ss_end_global
        ss_start_global = np.zeros([comm.size],dtype='int');
        ss_end_global = np.zeros([comm.size],dtype='int');

        # istock_start_global, istock_end_global
        istock_start_global = np.zeros([comm.size],dtype='int');
        istock_end_global = np.zeros([comm.size],dtype='int');

        # nslice_global, nstock_global
        nslice_global = np.zeros([comm.size],dtype='int');
        nstock_global = np.zeros([comm.size],dtype='int');

        # MN / nproc
        aa = (nstock * nslice) // comm.size;

        # Mod(MN, nproc)
        bb = (nstock * nslice) %  comm.size;

        for iproc in np.arange(comm.size):
            if iproc < bb :
                ss_start_global_temp = iproc * aa + iproc;
                ss_end_global_temp = (iproc + 1) * aa + (iproc + 1);
            else :
                ss_start_global_temp = iproc * aa + bb;
                ss_end_global_temp = (iproc + 1) * aa + bb;

            ss_start_global[iproc] = ss_start_global_temp;
            ss_end_global[iproc] = ss_end_global_temp;
            nslice_global[iproc] = ss_end_global_temp - ss_start_global_temp;

            istock_start_global_temp = (ss_start_global_temp) // nslice;
            istock_end_global_temp = (ss_end_global_temp - 1) // nslice + 1;
            istock_start_global[iproc] = istock_start_global_temp;
            istock_end_global[iproc] = istock_end_global_temp;
            nstock_global[iproc] = istock_end_global_temp - istock_start_global_temp;

        ######
        # Debug
        print("ss_start_global = ", ss_start_global);
        print("ss_end_global = ", ss_end_global);
        print("nslice_global = ", nslice_global);
        
        print("istock_start_global = ", istock_start_global);
        print("istock_end_global = ", istock_end_global);
        print("nstock_global = ", nstock_global);
        print("=======================");

    comm.Barrier();

    ######
    # For all procs
    ######
    # MN / nproc
    aa = (nstock * nslice) // comm.size;

    # Mod(MN, nproc)
    bb = (nstock * nslice) %  comm.size;

    if comm.rank < bb :
        ss_start_proc = comm.rank * aa + comm.rank;
        ss_end_proc = (comm.rank + 1) * aa + (comm.rank + 1);
    else :
        ss_start_proc = comm.rank * aa + bb;
        ss_end_proc = (comm.rank + 1) * aa + bb;

    nslice_proc = ss_end_proc - ss_start_proc;
    istock_start_proc = (ss_start_proc) // nslice;
    istock_end_proc = (ss_end_proc - 1) // nslice + 1;
    nstock_proc = istock_end_proc - istock_start_proc;

    ######
    # ROOT read in all the stock info
    if comm.rank == 0:

        price_global = np.zeros([nstock,npts-nval],dtype="float64");

        price_val_global = np.zeros([nval+tslen[-1]],dtype="float64");

        for istock in np.arange(nstock):
            # one_stock is a np.array([])
            print("Reading "+stock_name[istock]);
            stock_temp = io_price(stock_name[istock]);

            # price_init is in float64 type, price_init[1:npts]
            price_init = collapse(stock_temp, time_intval)[stock_offset[istock]:stock_offset[istock]+npts]

            ######
            # Divide prices into two, roughly equal sized, periods:
            price_global[istock,0:npts-nval] = price_init[0:npts-nval];

            # print("price_global[0:5] = ", price_global[istock,0:5]);

            ######
            # We will predict the price of the first stock
            if (istock == 0):
                price_val_global[0:nval+tslen[-1]] = price_init[(npts-nval-tslen[-1]):];

        # print("price_val_global[0:5] = ", price_val_global[0:5]);
        # Now we simply broadcast the total input data to all the proc  
    else:
        price_global = np.zeros([nstock,npts-nval],dtype="float64");
        price_val_global = np.zeros([nval+tslen[-1]],dtype="float64");

    ######
    # Broadcast all the data
    comm.Bcast(price_global,root=0);
    comm.Bcast(price_val_global,root=0);

    ######
    # print("proc = ", comm.rank, "price_global[0:5] = ", price_global[0,0:5]);
    # print("proc = ", comm.rank, "price_val_global[0:5] = ", price_val_global[0:5]);    
    
    if comm.rank == 0:
        T2 = MPI.Wtime()
        print("Time[ Read and distribute stock data ] : ", T2 - T1)

    ts_proc = np.zeros([nslice_proc],dtype="float64");
    
    # ######
    # # For all procs
    # # [TODO] fine-grained data parallel
    # ######
    # # Broadcast price_global, price_val
    # price_proc = np.zeros([nstock_proc,npts-nval],dtype="float");
    # price_val_proc = np.zeros([nval+tslen[-1]],dtype="float");
    # # Send 
    # if comm.rank == 0:
    #     comm.Isend()
    
    # ######
    # len1 = len(price)
    # len2 = len(price2)

    ######
    # Need debug here
    s_list_temp = np.zeros((nstock*nslice,ncenter,np.max(tslen)+1),dtype='float64');
    s_list = np.zeros((nstock*nslice,ncenter,np.max(tslen+1)),dtype='float64');
    
    for iss in np.arange(ss_start_proc, ss_end_proc):
    # for iss in np.arange(1):
        # numofts = len(price) - itslen - time_window + 1
        # ts_tsy[0:numofts-1,0:itslen];
        islice=iss%nslice;
        istock=iss//nslice;

        itslen=tslen[islice];

        ts_tsy = np.transpose(generate_timeseries_row(price_global[istock,:price_len//2], itslen, time_window));
        # print("proc = ", comm.rank, "istock = ", istock, "islice = ", islice, "itslen = ", itslen, "ts_tsy[0,0] = ", ts_tsy[0,0], "ts_tsy[0,itslen] = ", ts_tsy[0,itslen]);
        
        ######
        # centers1[0:n_clusters-1, 0:n_features-1]
        centers = find_cluster_centers_row(ts_tsy, nkmean);

        # print("proc = ", comm.rank, "istock = ", istock, "islice = ", islice, "itslen = ", itslen, "centers[0,0] = ", centers[0,0]);
        
        ######
        s_list_temp[iss, 0:ncenter, 0:itslen+1] = choose_effective_centers(centers, ncenter, m_top_discard);

        # print("proc = ", comm.rank, "istock = ", istock, "islice = ", islice, "itslen = ", itslen, "s_list[0,0] = ", s_list[islice,0,0]);

    if comm.rank == 0:
        T3a = MPI.Wtime();

        
    ######
    comm.Allreduce([s_list_temp, MPI.DOUBLE], [s_list, MPI.DOUBLE], op = MPI.SUM);

    if comm.rank == 0:
        T3b = MPI.Wtime();
        print("Time[ Allreduce ] : ", T3b - T3a);
    
    # ######
    # # find_cluster_centers will use KMeans, which involves a random number generator
    # # we set the random seed as : np.random.seed(23)
    # ######

    if comm.rank == 0:
        T3 = MPI.Wtime();
        print("Time[ Find centers ] : ", T3 - T2);

    # #exit(0)

    ######
    # [Parallelized]
    ## Dpi_r, Dp = linear_regression_vars(price[len1/2:], s1, s2, s3, s4, s5, s6, time_window)
    # print("Bayesian regression done.")

    ######
    # predict the first stock
    # w will be broadcast to every proc
    w = linear_regression_vars_find_w(price_global[0,price_len//2:], s_list, tslen, nstock, nslice, time_window)

    comm.Barrier();
    
    if comm.rank == 0:
        T4 = MPI.Wtime();
        print("Time[ Bayesian + linear regression ] : ", T4 - T3);
        print("Done training. Start Validation.");

    ######
    # here s_list, w are all global variables
    # Only ROOT holds the correct answer
    dps_val, pprice_val = predict_dps_parallel(price_val_global, s_list, w, tslen, nstock, nslice, time_window)

    if comm.rank == 0:
        T5 = MPI.Wtime();
        print("Time[ Validation ] : ", T5 - T4);
        print("Done training. Start Validation.");
    
    #    plt.plot(dps_val)
    if comm.rank == 0:
        dprice = diff_price(price_val_global);
        ######
        # ????
        # Why 20?
        ######
        mdprice = mean_price(dprice, 20);
        plt.plot(pprice_val);
        #    plt.plot(dprice)
        #plt.plot(mdprice)
        plt.plot(price_val_global);
        plt.show();

        plt.plot(dprice[tslen[-1]+time_window:]);
        plt.plot(dps_val);
        plt.show();

        # evaluation 1 -- original
        #    bank_balance = evaluate_performance(price, dps, tslen[-1], time_window, t=0.001, step=1)
        #    print("Final Balance:", bank_balance)

        # evaluation 2 -- track the change of total asset
        all_asset = evaluate_performance_asset(price_val_global, dps_val, tslen[-1], time_window, t=0.0001, step=time_window);

        plt.plot(all_asset/10.0);
        plt.plot(collapse(price_val_global[tslen[-1]:], time_window));
        plt.title('Total asset');
        plt.show();
        print('Start Asset: %f' % all_asset[0])
        print('Final Asset: %f' % all_asset[-1])
        # print("Weights", w)

    comm.Barrier()

    if comm.rank == 0:
        Ttotal = MPI.Wtime()
        print("Time[ Total ] : ", Ttotal - T1)

if __name__ == '__main__':

    time_intval = 1.0; # basic unit
    time_window = 5; # in time_intval unit

    npts = 20000;  # including nval, and the rest: half to find centers, half to do linear regression
    nval = 4000;

    # ts1len, ts2len, ts3len = 20, 60, 120;

    nstock = 5;

    stock_name=["AAPL","GOOG","FB","AMZN","IBM"];

    tslen = np.array([20, 60, 120, 240, 360]);
    nslice = len(tslen);

    offset = tslen[-1] + 25000;
    # stock_offset = np.array([offset,offset-tslen[-1],offset,offset,offset]); 
    stock_offset = np.array([offset,offset,offset,offset,offset]);
   
    nkmean, ncenter, m_top_discard = 100, 60, 2 # discard top m centers, because these are likely rare

    ######
    # TODO
    # size check
    if (len(stock_offset) != nstock):
        if comm.rank == 0:
            print("size of stock_offset wrong");
            exit(123);

    if (len(stock_name) != nstock):
        if comm.rank == 0:
            print("size of stock_name wrong");
            exit(124);

    price_len = npts-nval;
    
    if comm.rank == 0:
        if (nstock * nslice < comm.size):
            print("[ERROR] nstock * nslice < nproc!!!");
            exit(233);
        
        print("rank = ", comm.rank, "size = ", comm.size);
        print("Parameter list");
        print("npts = ", npts, "nval = ", nval);
        print("nstock = ", nstock, " nslice = ", nslice);
        print("stock_offset = ", stock_offset);
        print("stock_name = ", stock_name);
        print("tslen = ", tslen);
        #    print("ts1len = ", ts1len, "ts2len = ", ts2len, "ts3len = ", ts3len)
        print("nkmean = ", nkmean, "ncenter = ", ncenter, "m_top_discard = ", m_top_discard);
        print("=======================");

    main()
