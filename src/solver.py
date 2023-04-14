import numpy as np

def contract3(X, v):
    
    assert (len(X.shape) == 3) and (X.shape[0] == len(v))
    out = np.zeros(X.shape[1: ])
    
    if len(v) == 0: return out
    
    for ii in range(len(v)):
        out += v[ii] * X[ii, :, :]
    
    return out

def fw_step(A, x, b, eta, alpha = None): # x and b must be 2d np.array and columns
    
    Ax = A @ x
    half_grad = (Ax - b).T @ A + (eta * x).T
    i = half_grad.argmin()

    if alpha != None:
        x = x * (1 - alpha)
        x[i] = x[i] + alpha
        return x
    
    d_x = -x; d_x[i] = 1 - x[i]

    if all(d_x == 0): 
        return x
    
    d_err = A[:, i: i + 1] - Ax
    step = - (half_grad @ d_x)[0, 0] / ((d_err ** 2).sum() + eta * (d_x ** 2).sum())
    constrained_step = min(1, max(0, step))
    
    return x + constrained_step * d_x

def sc_weight_fw(
        Y, 
        zeta, 
        intercept = True, 
        lmbda = None, 
        min_decrease = 1e-3, 
        max_iter = 1000
):
    
    T0 = Y.shape[1] - 1
    N0 = Y.shape[0]

    if lmbda == None:
        lmbda = np.atleast_2d(np.repeat(1 / T0, T0)).T
    
    if intercept:
        Y = Y - Y.mean(axis = 0)

    t = 0
    vals = np.zeros(max_iter)
    A = Y[:, :T0].copy()
    b = Y[:, T0: T0 + 1].copy()
    eta = N0 * np.real(zeta ** 2)

    while (t < max_iter) and ((t < 2) or (vals[t - 2] - vals[t - 1] > min_decrease ** 2)):
        lambda_p = fw_step(A, lmbda, b, eta) # make sure to np.atleast_2d lmbda (needs to be column)
        lmbda = lambda_p
        err = Y[: N0, :] @ np.append(lmbda, [[-1]], axis = 0)
        vals[t] = np.real(zeta ** 2) * (lmbda ** 2).sum() + (err ** 2).sum() / N0
        t += 1
    
    return {"lambda": lmbda, "vals": vals}

def sc_weight_fw_covariates(
        Y, 
        X = None, 
        zeta_lambda = 0, 
        zeta_omega = 0, 
        lambda_intercept = True, 
        omega_intercept = True, 
        min_decrease = 1e-3, 
        max_iter = 1000, 
        lmbda = None, 
        omega = None, 
        beta = None, 
        update_lambda = True, 
        update_omega = True
):
    
    if X == None: X = np.zeros((0, ) + Y.shape)
    
    assert ((len(Y.shape) == 2) and (len(X.shape) == 3) and (Y.shape == X.shape[1:]) and np.all(np.isfinite(X)) and np.all(np.isfinite(Y)))

    T0 = Y.shape[1] - 1
    N0 = Y.shape[0] - 1

    if len(X.shape) == 2: X = X.reshape((1, ) + X.shape)
    if lmbda == None: lmbda = np.atleast_2d(np.repeat(1 / T0, T0)).T
    if omega == None: omega = np.atleast_2d(np.repeat(1 / N0, N0)).T
    if beta == None: beta = np.zeros((X.shape[0], 1))

    def update_weights(Y, lmbda, omega):

        if lambda_intercept:
            Y_lambda = Y[:N0, :].copy() - np.atleast_2d(Y[:N0, :].mean(axis = 1)).T
        else:
            Y_lambda = Y[:N0, :].copy()
        
        if update_lambda: lmbda = fw_step(Y_lambda[:, :T0], lmbda, Y_lambda[:, T0: T0 + 1], N0 * np.real(zeta_lambda ** 2))

        err_lambda = Y_lambda @ np.append(lmbda, [[-1]], axis = 0)
        
        if omega_intercept:
            Y_omega = Y[:, :T0].T.copy() - np.atleast_2d(Y[:, :T0].T.mean(axis = 1)).T
        else:
            Y_omega = Y[:, :T0].T.copy()
        
        if update_omega: omega = fw_step(Y_omega[:, :N0], omega, Y_omega[:, N0: N0 + 1], T0 * np.real(zeta_omega ** 2))

        err_omega = Y_omega @ np.append(omega, [[-1]], axis = 0)

        val = np.real(zeta_omega ** 2) * (omega ** 2).sum() + np.real(zeta_lambda ** 2) * (lmbda ** 2).sum() + (err_omega ** 2).sum() / T0 + (err_lambda ** 2).sum() / N0

        return {"val": val, "lmbda": lmbda, "omega": omega, "err_lambda": err_lambda, "err_omega": err_omega}
    
    vals = np.zeros(max_iter)
    t = 0
    Y_beta = Y.copy() - contract3(X, beta)
    weights = update_weights(Y_beta, lmbda, omega)

    while (t < max_iter) and ((t < 2) or (vals[t - 2] - vals[t - 1] > min_decrease ** 2)):
        
        if X.shape[0] == 0: 
            grad_beta = np.zeros((0, 0)) 
        else:
            grad_beta = weights["err_lambda"].T @ X[:, :N0, :] @ weights["lmbda"] / N0 + weights["err_omega"].T @ X[:, :, :T0] @ weights["omega"] / T0
        
        alpha = 1 / t
        beta = beta - alpha * grad_beta
        Y_beta = Y.copy() - contract3(X, beta)
        weights = update_weights(Y_beta, weights["lmbda"], omega["omega"])
        vals[t] = weights["val"]
        t += 1
    
    return {"lmbda": weights["lmbda"], "omega": weights["omega"], "beta": beta, "vals": vals}

