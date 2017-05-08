# This module includes all the basic data preprocessing functions:
#	--io_price: read in csv price data
#	--diff_price: calculate the price change
# 	--mean_price: calculate the average price
#	--fillna: replace -9999 in data with zeros
#	--collapse: resample(undersample) the price data
#	--normsqr: calculate the square of l2 norm
#	--generate_timeseries: generate the time series with specified length and
# 		prediction time window
#
# Update: 04/19/2017

import pandas as pd
import numpy as np
from scipy import io
import math

def io_price(symbol):
	df = pd.read_csv("./data/{}.csv".format(symbol))
	return np.array(df['Close'])

def diff_price(price):
	dprice = np.zeros(len(price)-1, dtype=float)
	dprice[:] = price[1:] - price[:-1]
	return dprice

def smooth_price(price, window):
        newprice = np.zeros(len(price)-window, dtype=float)
        for i in range(len(price)-window):
                newprice[i] = np.mean(price[i:i+window])
        return newprice

def inv_diff_price(dprice, price0):
	price = np.zeros(len(dprice)+1, dtype=float)
	price[0] = price0
	for i in range(len(dprice)):
		price[i+1] = dprice[i] + price[i]
	return price

def mean_price(price, step):
	mprice = np.zeros(int(len(price)/step), dtype=float)
	for i in range(len(mprice)-1):
		mprice[i] = np.mean(price[i*step:(i+1)*step])
	return mprice

def fillna(price):
	if price[0] == -9999.0:
		price[0] = 0.0
	for i in range(len(price)-1):
		if price[i+1] == -9999.0:
			price[i+1] = price[i]
	return price

def collapse(price, time_intval):
	size = int(len(price)/time_intval)
	newprice = np.zeros(size, dtype=float)
	for i in range(size):
		newprice[i] = price[i*int(time_intval)]
	return newprice

def normsqr(v):
	return np.dot(v,v)

def generate_timeseries(price, ts_len, time_window):
	m = len(price) - ts_len - time_window + 1

	ts = np.zeros((m, ts_len), dtype=float)
	tsy = np.zeros(m, dtype=float)
	for i in range(m):
		ts[i, :ts_len] = price[i:i + ts_len]
		tsy[i] = price[i + ts_len + time_window -1] - price[i + ts_len - 1]
	return (ts, tsy)

def generate_timeseries_row(price, ts_len, time_window):
	m = len(price) - ts_len - time_window + 1;

	# ts= np.zeros((ts_len, m), dtype=float)
	# tsy = np.zeros(m, dtype=float)
	ts_tsy = np.zeros((ts_len+1, m), dtype=float);
	for i in range(m):
		ts_tsy[:ts_len, i] = price[i :i + ts_len];
		ts_tsy[ts_len,i] = price[i + ts_len + time_window -1] - price[i + ts_len - 1];
	return ts_tsy
