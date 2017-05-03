# This module contains the evaluation functions of the prediction model
# Update: 04/19/2017

import numpy as np

def evaluate_performance(price, dps, tslen_end, time_window, t, step):
    bank_balance = 0
    position = 0
    for i in range(tslen_end, len(price) - time_window, step):
        # long position - BUY
        if dps[i - tslen_end] > t and position <= 0:
            position += 1
            bank_balance -= price[i]
        # short position - SELL
        if dps[i - tslen_end] < -t and position >= 0:
            position -= 1
            bank_balance += price[i]
    # sell what you bought
    if position == 1:
        bank_balance += price[len(price) - 1]
    # pay back what you borrowed
    if position == -1:
        bank_balance -= price[len(price) - 1]
    return bank_balance

def evaluate_performance_asset(price, dps, tslen_end, time_window, t, step):
    nShare = 0
    cash_balance = price[tslen_end]*10.0
    all_asset = []

    # print("evaluate_steps: tslen_end = ", tslen_end, " len(price) - time_window = ", len(price) - time_window, " step = ", step);
    
    for i in range(tslen_end, len(price) - time_window, step):
        # long position - BUY
        if dps[i - tslen_end] > t and cash_balance > 0:
            nBuy = np.floor(cash_balance/price[i])
            nShare += nBuy
            # if nBuy > 0:
                # print('Time ', i, 'Buy , ', nBuy, ' #Share, ', nShare)
            cash_balance -= price[i]*nBuy
        # short position - SELL
        if dps[i - tslen_end] < -t and nShare > 0:
            nSell = nShare
            cash_balance += price[i] * nSell
            nShare = 0
            # if nSell > 0:
                # print('Time ', i, 'Sell, ', nSell, ' #Share, ', nShare)
        all_asset.append(cash_balance + nShare*price[i])
    return np.array(all_asset)
