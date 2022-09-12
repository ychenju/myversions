# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# return the index of smallest positive value of a 3x3 matrix
def isp3(eV):
    param = [1,1,1]
    for i, x in enumerate(eV):
        if x < 0:
            param[i] = 0
        if x > eV[(i+1)%3] and eV[(i+1)%3] >= 0:
            param[i] = 0
        if x > eV[(i-1)%3] and eV[(i-1)%3] >= 0:
            param[i] = 0
    return param.index(1)

# plane fitting
def fit(X, Y, T):
    X_ = X.flatten()
    Y_ = Y.flatten()
    T_ = T.flatten()

    M_ = np.array([[np.nanmean(X_*X_)-np.nanmean(X_)**2, np.nanmean(X_*Y_)-np.nanmean(X_)*np.nanmean(Y_), np.nanmean(X_*T_)-np.nanmean(X_)*np.nanmean(T_)],
                   [np.nanmean(X_*Y_)-np.nanmean(X_)*np.nanmean(Y_), np.nanmean(Y_*Y_)-np.nanmean(Y_)**2, np.nanmean(Y_*T_)-np.nanmean(Y_)*np.nanmean(T_)],
                   [np.nanmean(X_*T_)-np.nanmean(X_)*np.nanmean(T_), np.nanmean(Y_*T_)-np.nanmean(Y_)*np.nanmean(T_), np.nanmean(T_*T_)-np.nanmean(T_)**2]])

    eValue, eVector = np.linalg.eig(M_)

    fVec = eVector.T[isp3(eValue)]
    A0 = -fVec[0]/fVec[2]
    B0 = -fVec[1]/fVec[2]
    T0 = -A0*np.nanmean(X_) - B0*np.nanmean(Y_) + np.nanmean(T_)

    T1 = T0 + A0*X + B0*Y
    Tr = T - T1       

    return T1, Tr

def rmse(T):
    T_ = T.flatten()
    return np.sqrt(np.nanmean(T_*T_))

def sigma(X, Y, T):
    _, Tr = fit(X,Y,T)
    return rmse(Tr)

def mean3(a2d):
    a3_00 = a2d[:-2:3,:-2:3]
    a3_01 = a2d[:-2:3,1:-1:3]
    a3_02 = a2d[:-2:3,2::3]
    a3_10 = a2d[1:-1:3,:-2:3]
    a3_11 = a2d[1:-1:3,1:-1:3]
    a3_12 = a2d[1:-1:3,2::3]
    a3_20 = a2d[2::3,:-2:3]
    a3_21 = a2d[2::3,1:-1:3]
    a3_22 = a2d[2::3,2::3]
    a3_m = (a3_00+a3_01+a3_02+a3_10+a3_11+a3_12+a3_20+a3_21+a3_22)/9.
    return a3_m
