import numpy as np
from . import utils
from . import solver

def sparsify_function(v):

    v[v <= max(v) / 4] = 0
    
    return v / v.sum()

def synthdid_estimate(
        Y, N0, T0, X = None, 
        noise_level = None, 
        eta_omega = None, eta_lambda = 1e-6, 
        zeta_omega = None, zeta_lambda = None, 
        omega_intercept = True, lambda_intercept = True, 
        weights = {"omega": None, "lmbda": None}, 
        update_omega = None, update_lambda = None, 
        min_decrease = None, max_iter = 1e4, 
        sparsify = sparsify_function, 
        max_iter_pre_sparsify = 100

):
    
    if X == None: X = X = np.zeros((0, ) + Y.shape)
    if noise_level == None: noise_level = np.std(np.diff(Y[:N0, :T0]))
    if eta_omega == None: eta_omega = ((Y.shape[0] - N0) * (Y.shape[1] - T0)) ** (1 / 4)
    if zeta_omega == None: zeta_omega = eta_omega * noise_level
    if zeta_lambda == None: zeta_lambda = eta_lambda * noise_level
    if update_omega == None: update_omega = (weights["omega"] == None)
    if update_lambda == None: update_lambda = (weights["lmbda"] == None)
    if min_decrease == None: min_decrease = 1e-5 * noise_level

    assert ((Y.shape[0] > N0) and (Y.shape[1] > T0) and (X.ndim in (2, 3)) and (X.shape[1:] == Y.shape) and (type(weights) == dict))
    assert (((weights["lmbda"] == None) or (weights["lmbda"].shape[0] == T0)) and ((weights["omega"] == None) or (weights["omega"].shape[0] == N0)))
    assert (((weights["lmbda"] != None) or (update_lambda)) and ((weights["omega"] != None) or (update_omega)))

    if X.ndim == 2: X = X.reshape((1, ) + X.shape)
    if sparsify == None: max_iter_pre_sparsify = max_iter

    N1 = Y.shape[0] - N0
    T1 = Y.shape[1] - T0

    Yc = utils.collapsed_form(Y, N0, T0)

    if X.shape[0] == 0:
        weights["vals"] = None
        weights["lambda_vals"] = None
        weights["omega_vals"] = None

        if update_lambda:
            lambda_opt = solver.sc_weight_fw(Yc[:N0, :], zeta=zeta_lambda, intercept=lambda_intercept, lmbda=weights["lmbda"], 
                                             min_decrease=min_decrease, max_iter=max_iter)
            
            if sparsify != None:
                lambda_opt = solver.sc_weight_fw(Yc[:N0, :], zeta=zeta_lambda, intercept=lambda_intercept, lmbda=sparsify(lambda_opt["lmbda"]), 
                                                 min_decrease=min_decrease, max_iter=max_iter)
            
            weights["lmbda"] = lambda_opt["lmbda"]
            weights["lambda_vals"] = lambda_opt["vals"]
            weights["vals"] = lambda_opt["vals"]
        
        if update_omega:
            omega_opt = solver.sc_weight_fw(Yc[:, :T0].T, zeta=zeta_lambda, intercept=lambda_intercept, lmbda=weights["lmbda"], 
                                            min_decrease=min_decrease, max_iter=max_iter)
            
            if sparsify != None:
                omega_opt = solver.sc_weight_fw(Yc[:, :T0].T, zeta=zeta_lambda, intercept=lambda_intercept, lmbda=weights["lmbda"], 
                                                min_decrease=min_decrease, max_iter=max_iter)
                
            weights["omega"] = omega_opt["omega"]
            weights["omega_vals"] = omega_opt["vals"]
            if weights["vals"] == None:
                weights["vals"] = lambda_opt["vals"]
            else:
                weights["vals"] = utils.pairwise_sum_decreasing(weights["vals"], omega_opt["vals"])
            
    else:
        Xc = np.array([utils.collapsed_form(x, N0, T0) for x in X])
        weights = solver.sc_weight_fw_covariates(Yc, Xc, zeta_lambda=zeta_lambda, zeta_omega=zeta_omega, 
                                                 lambda_intercept=lambda_intercept, omega_intercept=omega_intercept, 
                                                 min_decrease=min_decrease, max_iter=max_iter, 
                                                 lmbda=weights["lmbda"], omega=weights["omega"], update_lambda=update_lambda, update_omega=update_omega)
        
    X_beta = solver.contract3(X, weights["beta"])
    o_hat = - np.append(weights["omega"], np.array([[1 / N1] for i in range(N1)]), axis = 0).T
    l_hat = - np.append(weights["lmbda"], np.array([[1 / T1] for i in range(T1)]), axis = 0)
    estimate = o_hat @ (Y - X_beta) @ l_hat

    return estimate

def sc_estimate(Y, N0, T0, eta_omega = 1e-6, **kwargs):
    estimate = synthdid_estimate(Y, N0, T0, eta_omega = eta_omega, 
                                 weights = {"lmbda": np.zeros((T0, 1))}, omega_intercept=False, **kwargs)
    return estimate

def did_estimate(Y, N0, T0, **kwargs):
    estimate = synthdid_estimate(Y, N0, T0, weights = {"lmbda": np.zeros((T0, 1)) + 1 / T0, "omega": np.zeros((N0, 1)) + 1 / N0}, **kwargs)

    return estimate

