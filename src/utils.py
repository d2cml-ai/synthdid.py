import numpy as np

def collapsed_form(Y, N0, T0): 

    N = Y.shape[0]; T = Y.shape[1]

    Y_T0N0 = np.atleast_2d(Y[:N0, :T0])
    Y_T1N0 = np.atleast_2d(Y[:N0, T0: T].mean(axis = 1)).T
    Y_T0N1 = np.atleast_2d(Y[N0: N, :T0].mean(axis = 0))
    Y_T1N1 = np.atleast_2d(Y[N0: N, T0:T].mean())

    Yc = np.append(np.append(Y_T0N0, Y_T1N0, axis = 1), np.append(Y_T0N1, Y_T1N1, axis = 1), axis = 0)

    return Yc

def pairwise_sum_decreasing(x, y):

    na_x = np.isnan(x)
    na_y = np.isnan(y)
    x[np.isnan(x)] = x[np.logical_not(na_x)].min()
    y[np.isnan(y)] = y[np.logical_not(na_y)].min()
    pairwise_sum = x + y
    pairwise_sum[np.logical_and(na_x, na_y)] = np.nan

    return pairwise_sum

def random_low_rank():

    n_0 = 100
    n_1 = 10
    T_0 = 120
    T_1 = 20
    n = n_0 + n_1
    T = T_0 + T_1
    tau = 1
    sigma = 0.5
    rank = 2
    rho = 0.7
    var = rho ** abs((np.arange(T) + 1)[:, None] - (np.arange(T) + 1))

    W = np.atleast_2d(np.arange(n) + 1 > n_0).TW @ np.atleast_2d((np.arange(T) + 1) > T_0)
    

