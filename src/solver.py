from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_slsqp
from toolz import partial
from sklearn.model_selection import KFold, TimeSeriesSplit, RepeatedKFold
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from bayes_opt import BayesianOptimization

# SDID

def est_zeta(n_treat, n_post_term, Y_pre_c):

    return (n_treat * n_post_term) ** (1 / 4) * np.std(Y_pre_c.diff().dropna().values)

def est_omega(l2_loss, Y_pre_c, Y_pre_t, zeta):

    Y_pre_t = Y_pre_t.copy()
    n_features = Y_pre_t.shape[1]
    nrow = Y_pre_t.shape[0]

    _w = np.repeat(1 / n_features, n_features)
    _w0 = 1

    start_w = np.append(_w, _w0)

    if type(Y_pre_t) == pd.core.frame.DataFrame:
        Y_pre_t = Y_pre_t.mean(axis=1)

    max_bnd = abs(Y_pre_t.mean()) * 2
    w_bnds = tuple(
        (0, 1) if i < n_features else (max_bnd * -1, max_bnd) 
        for i in range(n_features + 1)
    )

    caled_w = fmin_slsqp(
        partial(
            l2_loss, 
            X = Y_pre_c, 
            y = Y_pre_t, 
            zeta = zeta, 
            nrow = nrow
        ), 
        start_w, 
        f_eqcons = lambda x: np.sum(x[:n_features]) - 1, 
        bounds = w_bnds, 
        disp = False
    )
    
    return caled_w

def est_lambda(l2_loss, Y_pre_c, Y_post_c):

    Y_pre_c_T = Y_pre_c.T
    Y_post_c_T = Y_post_c.T

    n_pre_term = Y_pre_c_T.shape[1]

    _lambda = np.repeat(1 / n_pre_term, n_pre_term)
    _lambda0 = 1

    start_lambda = np.append(_lambda, _lambda0)

    if type(Y_post_c_T) == pd.core.frame.DataFrame:
        Y_post_c_T = Y_post_c_T.mean(axis=1)
    
    max_bnd = abs(Y_post_c_T.mean()) * 2
    lambda_bnds = tuple(
        (0, 1) if i < n_pre_term else (max_bnd * -1, max_bnd)
        for i in range(n_pre_term + 1)
    )

    caled_lambda = fmin_slsqp(
        partial(
            l2_loss, 
            X = Y_pre_c_T, 
            y = Y_post_c_T, 
            zeta = 0, 
            nrow = 0
        ), 
        start_lambda, 
        f_eqcons = lambda x: np.sum(x[:n_pre_term]) - 1,
        bounds = lambda_bnds, 
        disp = False
    )

    return caled_lambda[:n_pre_term]

def l2_loss(W, X, y, zeta, nrow):

    if type(y) == pd.core.frame.DataFrame:
        y = y.mean(axis = 1)
    
    _X = X.copy()
    _X["intercept"] = 1

    return np.sum((y - _X.dot(W)) ** 2) + nrow * zeta ** 2 * np.sum(W[: -1] ** 2)

# SC

# CV search for zeta

# Sparce estimation for omega
