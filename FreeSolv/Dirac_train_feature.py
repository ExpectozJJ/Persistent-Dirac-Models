import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import scipy as sp
from math import sqrt
import pickle
import sys 
import os 
import pandas as pd
import math
from xgboost import XGBRegressor

#n_estimators, lr, depth, subsample, mss = 20000, 0.05, 7, 0.4, 3

X = np.load("./X.npy", allow_pickle=True)
Y = np.load("./solv.npy", allow_pickle=True)

print(np.shape(X))
#print(np.shape(Y))

ratio = 0.1*10/9

results = []
for idx in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=ratio)

    #reg = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate=lr, validation_fraction=ratio, n_iter_no_change=5, tol=0.01, max_features='sqrt', random_state=idx, max_depth=depth, subsample=subsample, min_samples_split=mss)
    #reg.fit(X_train, y_train)
    reg = XGBRegressor(n_estimators=20000, eta=0.1, max_depth=7, subsample=0.4, colsample_bytree=0.8)
    reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric ='rmse', early_stopping_rounds=25, verbose=False)

    y_pred = reg.predict(X_test)
    pearcorr =  sp.stats.pearsonr(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    print(pearcorr[0], rmse)
    results.append([pearcorr[0], rmse])

print(np.mean(results, axis=0), np.std(results, axis=0))
