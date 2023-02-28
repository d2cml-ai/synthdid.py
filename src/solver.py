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


## est_omega and est_lambda together are equivalent to the fw.step and sc.weight.fw functions in the R package
def est_omega(l2_loss, Y_pre_c, Y_pre_t, zeta):

    Y_pre_t = Y_pre_t.copy()
    n_features = Y_pre_c.shape[1]
    nrow = Y_pre_c.shape[0]

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

## loss function, equivalent to what is included in fw.step and sc.weights.fw in the R package
def l2_loss(W, X, y, zeta, nrow):

    if type(y) == pd.core.frame.DataFrame:
        y = y.mean(axis = 1)
    
    _X = X.copy()
    _X["intercept"] = 1

    return np.sum((y - _X.dot(W)) ** 2) + nrow * zeta ** 2 * np.sum(W[: -1] ** 2)

# SC

## rmse_loss and rmse_loss_with_v are loss functinos for estimations. Equivalent to what is included in fw.step, sc.weight.fw, and sc.weight.fw.covariates in the R package
def rmse_loss(W, X, y, intercept = True) -> float:

    if type(y) == pd.core.frame.DataFrame:
        y = y.mean(axis=1)

    _X = X.copy()

    if intercept:
        _X["intercept"] = 1

    return np.mean(np.sqrt((y - _X.dot(W)) ** 2))

def rmse_loss_with_V(W, V, X, y) -> float:
    if type(y) == pd.core.frame.DataFrame:
        y = y.mean(axis=1)
    _rss = (y - X.dot(W)) ** 2

    _n = len(y)
    _importance = np.zeros((_n, _n))

    np.fill_diagonal(_importance, V)

    return np.sum(_importance @ _rss)

## _v_loss and estimate_v together are equivalent to the sc.weight.fw.covariates function in the R package
def _v_loss(V, X, y, Y_pre_t, Y_pre_c, return_loss=True):
    
    Y_pre_t = Y_pre_t.copy()

    n_features = Y_pre_c.shape[1]
    _w = np.repeat(1 / n_features, n_features)

    if type(Y_pre_t) == pd.core.frame.DataFrame:
        Y_pre_t = Y_pre_t.mean(axis=1)

    w_bnds = tuple((0, 1) for i in range(n_features))
    _caled_w = fmin_slsqp(
        partial(rmse_loss_with_V, V=V, X=X, y=y),
        _w,
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=w_bnds,
        disp=False,
    )
    if return_loss:
        return rmse_loss(_caled_w, Y_pre_c, Y_pre_t, intercept = False)
    else:
        return _caled_w

def estimate_v(additional_X, additional_y, Y_pre_t, Y_pre_c):
    _len = len(additional_X)
    _v = np.repeat(1 / _len, _len)

    caled_v = fmin_slsqp(
        partial(
        _v_loss, 
        X = additional_X, 
        y = additional_y, 
        Y_pre_t = Y_pre_t, 
        Y_pre_c = Y_pre_c
        ),
        _v,
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=tuple((0, 1) for i in range(_len)),
        disp=False,
    )
    return caled_v

# This function is equivalent to fw.step, sc.weights.fw, and sc.weight.fw.covariates in the R package
def est_omega_ADH(
    Y_pre_c, Y_pre_t, additional_X = pd.DataFrame(), additional_y = pd.DataFrame()
):
    """
    # SC
    estimating omega for synthetic control method (not for synthetic diff.-in-diff.)
    """
    Y_pre_t = Y_pre_t.copy()

    n_features = Y_pre_c.shape[1]
    nrow = Y_pre_c.shape[0]

    _w = np.repeat(1 / n_features, n_features)

    if type(Y_pre_t) == pd.core.frame.DataFrame:
        Y_pre_t = Y_pre_t.mean(axis=1)

    # Required to have non negative values
    w_bnds = tuple((0, 1) for i in range(n_features))

    if len(additional_X) == 0:
        caled_w = fmin_slsqp(
            partial(rmse_loss, X = Y_pre_c, y = Y_pre_t, intersept=False),
            _w,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=w_bnds,
            disp=False,
        )

        return caled_w
    else:
        assert additional_X.shape[1] == Y_pre_c.shape[1]
        if type(additional_y) == pd.core.frame.DataFrame:
            additional_y = additional_y.mean(axis=1)

        # normalized
        temp_df = pd.concat([additional_X, additional_y], axis=1)
        ss = StandardScaler()
        ss_df = pd.DataFrame(
            ss.fit_transform(temp_df), columns=temp_df.columns, index=temp_df.index
        )

        ss_X = ss_df.iloc[:, :-1]
        ss_y = ss_df.iloc[:, -1]

        add_X = pd.concat([Y_pre_c, ss_X])
        add_y = pd.concat([Y_pre_t, ss_y])

        caled_v = estimate_v(additional_X = add_X, additional_y = add_y, Y_pre_t = Y_pre_t, Y_pre_c = Y_pre_c)

        return _v_loss(caled_v, X = add_X, y = add_y, Y_pre_t = Y_pre_t, Y_pre_c = Y_pre_c, return_loss=False)
    


# CV search for zeta

# Sparce estimation for omega
