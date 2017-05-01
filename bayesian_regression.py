# The module of bayesian regression based price prediction model
#
# Update: 04/19/2017


import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cluster import KMeans
import pandas as pd

######
# MPI
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
comm = MPI.COMM_WORLD

np.random.seed(23)

from basic_processing import *

def find_cluster_centers(ts, tsy, k):
        timeseries = np.concatenate((ts, np.reshape(tsy, (len(tsy), 1))), axis=1)
        k_means = KMeans(n_clusters=k)
        k_means.fit(timeseries)
        return k_means.cluster_centers_


def choose_effective_centers(centers, n, m):
        return centers[np.argsort(np.ptp(centers, axis=1))[-n-m:len(centers)-m]]

def predict_dpi(x, centers, lnorm):
        num = 0.0
        den = 0.0
        x = x - np.mean(x)

        for i in range(len(centers)):
                y_i = centers[i, len(x)]
                x_i = centers[i, :len(x)] - np.mean(centers[i, :len(x)])
                if lnorm == 2:
                        exp = np.exp(-0.25 * normsqr(x - x_i))
                elif lnorm == 1:
                        exp = np.exp(-0.25 * np.sum(np.abs(x - x_i)))
                num += y_i * exp
                den += exp
        return num / (den + 1.0e-5)

def linear_regression_vars(price, s1, s2, s3, s4, s5, s6, time_window):
        print("check", len(s1), len(s2), len(s3), len(s4), len(s5), len(s6))
        maxlen = np.max(np.array([len(s1[0])-1, len(s2[0])-1, len(s3[0])-1]))

        ######
        # Use one XY = np.zeros((len(price) - maxlen - time_window, 7), dtype=float)
        ######

        X = np.zeros((len(price) - maxlen - time_window, 6), dtype=float)
        Y = np.zeros(len(price) - maxlen - time_window, dtype=float)

        XY0 = np.zeros((len(price) - maxlen - time_window, 7), dtype=float)
        XY = np.zeros((len(price) - maxlen - time_window, 7), dtype=float)
        
        ###################################        
        # TODO
        # Add parallelism here
        ###################################
        # i = maxlen, maxlen+1, maxlen+2, ..., len(price) - time_window - 1
        # Separate the range across the procs
        ###################################
        # for i in range(maxlen, len(price) - time_window):

        # Define proc-dependent range

        offset = maxlen;
        work_global = len(price) - time_window - maxlen;
        
        work_proc = work_global // comm.size; # integer division
        work_residue = work_global - work_proc * comm.size;

        if (comm.rank < work_residue) :
                work_proc = work_proc + 1;

        start_proc = work_proc * comm.rank + offset
        
        if (comm.rank >= work_residue) :
                start_proc = start_proc + work_residue;

        # Because we use C convention, end_proc - start_proc = work_proc
        end_proc = start_proc + work_proc

        ######
        # check the ranges
        # print("proc = ", comm.rank, "size = ", comm.size, "offset = ", offset, " work_global = ", work_global, " work_proc = ", work_proc, " start_proc = ", start_proc, " end_proc = ", end_proc)

        for i in range(start_proc, end_proc):        
              dp = price[i + time_window] - price[i]

              dp1 = predict_dpi(price[i - len(s1[0]) + 1:i], s1, lnorm=2)
              dp2 = predict_dpi(price[i - len(s2[0]) + 1:i], s2, lnorm=2)
              dp3 = predict_dpi(price[i - len(s3[0]) + 1:i], s3, lnorm=2)

              dp4 = predict_dpi(price[i - len(s4[0]) + 1:i], s4, lnorm=2)
              dp5 = predict_dpi(price[i - len(s5[0]) + 1:i], s5, lnorm=2)
              dp6 = predict_dpi(price[i - len(s6[0]) + 1:i], s6, lnorm=2)

              # # i - maxlen = 0, 1, 2, ...
              # X[i - maxlen, :] = [dp1, dp2, dp3, dp4, dp5, dp6]
              # Y[i - maxlen] = dp

              # i - maxlen = 0, 1, 2, ...
              XY0[i - maxlen, :] = [dp1, dp2, dp3, dp4, dp5, dp6, dp]

        # ######
        # # MPI reduce sum here, allreduce(self, sendbuff, recvbuff, op)
        comm.Allreduce([XY0,MPI.DOUBLE],[XY, MPI.DOUBLE], op=MPI.SUM)

        # print("proc = ", comm.rank, " XY[0,:]", XY[0,:]);
        ######
        # return X, Y

        return XY[:,0:6], XY[:,6]

###################################

def linear_regression_vars_find_w(price, s_list, nstock, nts, time_window):
        # print("check", len(s1), len(s2), len(s3), len(s4), len(s5), len(s6))
        # maxlen = np.max(np.array([len(s1[0])-1, len(s2[0])-1, len(s3[0])-1]))
        maxlen = np.max(np.array(s_list[0:nts]))
        
        ######
        # Use one XY = np.zeros((len(price) - maxlen - time_window, 7), dtype=float)
        ######

        X = np.zeros((len(price) - maxlen - time_window, 6), dtype=float)
        Y = np.zeros(len(price) - maxlen - time_window, dtype=float)

        XY0 = np.zeros((len(price) - maxlen - time_window, 7), dtype=float)
        XY = np.zeros((len(price) - maxlen - time_window, 7), dtype=float)
        
        ###################################        
        # TODO
        # Add parallelism here
        ###################################
        # i = maxlen, maxlen+1, maxlen+2, ..., len(price) - time_window - 1
        # Separate the range across the procs
        ###################################
        # for i in range(maxlen, len(price) - time_window):

        # Define proc-dependent range
        offset = maxlen;
        work_global = len(price) - time_window - maxlen;
        
        work_proc = work_global // comm.size; # integer division
        work_residue = work_global - work_proc * comm.size;

        if (comm.rank < work_residue) :
                work_proc = work_proc + 1;

        start_proc = work_proc * comm.rank + offset
        
        if (comm.rank >= work_residue) :
                start_proc = start_proc + work_residue;

        # Because we use C convention, end_proc - start_proc = work_proc
        end_proc = start_proc + work_proc

        ######
        # check the ranges
        # print("proc = ", comm.rank, "size = ", comm.size, "offset = ", offset, " work_global = ", work_global, " work_proc = ", work_proc, " start_proc = ", start_proc, " end_proc = ", end_proc)

        for i in range(start_proc, end_proc):        
              dp = price[i + time_window] - price[i]

              dp1 = predict_dpi(price[i - len(s1[0]) + 1:i], s1, lnorm=2)
              dp2 = predict_dpi(price[i - len(s2[0]) + 1:i], s2, lnorm=2)
              dp3 = predict_dpi(price[i - len(s3[0]) + 1:i], s3, lnorm=2)

              dp4 = predict_dpi(price[i - len(s4[0]) + 1:i], s4, lnorm=2)
              dp5 = predict_dpi(price[i - len(s5[0]) + 1:i], s5, lnorm=2)
              dp6 = predict_dpi(price[i - len(s6[0]) + 1:i], s6, lnorm=2)

              # # i - maxlen = 0, 1, 2, ...
              # X[i - maxlen, :] = [dp1, dp2, dp3, dp4, dp5, dp6]
              # Y[i - maxlen] = dp

              # i - maxlen = 0, 1, 2, ...
              XY0[i - maxlen, :] = [dp1, dp2, dp3, dp4, dp5, dp6, dp]

        # ######
        # # MPI reduce sum here, allreduce(self, sendbuff, recvbuff, op)
        comm.Reduce([XY0,MPI.DOUBLE],[XY, MPI.DOUBLE], op=MPI.SUM, root = 0)

        ######
        if comm.rank == 0:
                ######
                # datatype and length of w : double, w.shape = (7)
                w = find_w_linear_regression(XY[:,0:6], XY[:,6])

        else :
                w = np.zeros((7),dtype='float64');

        ## Broadcast w
        comm.Bcast([w,MPI.DOUBLE], root = 0)
        
        # print("proc = ", comm.rank, " XY[0,:]", XY[0,:]);
        ######
        # return XY[:,0:6], XY[:,6]

        return w

###################################

def find_w_linear_regression(X, Y):
        clf = linear_model.LinearRegression()
#       clf = linear_model.Lasso(alpha = 0.01)
        clf.fit(X, Y)
        # w0 = clf.intercept_
        # w1, w2, w3, w4, w5, w6 = clf.coef_
        w = np.zeros(7,dtype="float64");
        w[0] = clf.intercept_;
        w[1:7] = clf.coef_.copy();
        # return w0, w1, w2, w3, w4, w5, w6
        return w

def predict_dps(price, s1, s2, s3, s4, s5, s6, w, time_window):
        maxlen = np.max(np.array([len(s1[0])-1, len(s2[0])-1, len(s3[0])-1]))
        dps = []
        pprice = np.zeros(len(price))
        w0, w1, w2, w3, w4, w5, w6 = w

        ######
        # TODO
        for i in range(maxlen, len(price) - time_window):

        # # Define proc-dependent range
        # offset = maxlen;
        # work_global = len(price) - time_window - maxlen;
        
        # work_proc = work_global // comm.size; # integer division
        # work_residue = work_global - work_proc * comm.size;

        # if (comm.rank < work_residue) :
        #         work_proc = work_proc + 1;

        # start_proc = work_proc * comm.rank + offset
        
        # if (comm.rank >= work_residue) :
        #         start_proc = start_proc + work_residue;

        # # Because we use C convention, end_proc - start_proc = work_proc
        # end_proc = start_proc + work_proc

        # # for i in range(start_proc, end_proc):
                print("in predict_dps", i);
                dp1 = predict_dpi(price[i - len(s1[0]) + 1:i], s1, lnorm=2)
                dp2 = predict_dpi(price[i - len(s2[0]) + 1:i], s2, lnorm=2)
                dp3 = predict_dpi(price[i - len(s3[0]) + 1:i], s3, lnorm=2)

                dp4 = predict_dpi(price[i - len(s4[0]) + 1:i], s4, lnorm=2)
                dp5 = predict_dpi(price[i - len(s5[0]) + 1:i], s5, lnorm=2)
                dp6 = predict_dpi(price[i - len(s6[0]) + 1:i], s6, lnorm=2)


                dp = w0 + w1 * dp1 + w2 * dp2 + w3 * dp3 + w4 * dp4 + w5 * dp5 + w6 * dp6
                dp *= 1.0
                dps.append(float(dp))
                pprice[i + time_window] = float(dp) + price[i]

        pprice[:maxlen + time_window] =  pprice[maxlen + time_window]

        return np.array(dps), pprice
