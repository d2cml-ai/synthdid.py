import numpy as np, pandas as pd

def fw_step(A, b, x, eta, alpha=None):
    x = np.array(x)
    Ax = np.dot(A, x)
    half_grad = np.dot((Ax - b), A) + eta * np.transpose(x)
    i = np.argmin(half_grad)
    if alpha is not None:
        x *= (1 - alpha)
        x[i] += alpha
        return x
    else:
        d_x = x.copy() * -1
        d_x[i] = 1 - x[i]
        if np.all(d_x == 0):
            return x
        d_err = A[:, i] - Ax
        step_upper = np.dot(-half_grad,  d_x)
        step_bot = np.sum(d_err ** 2) + eta * np.sum(d_x ** 2)
        step = step_upper / step_bot
        constrained_step = np.min([1, np.max([0, step])])
        return x + constrained_step * d_x


# x -> lambda(r)
def sc_weight_fw(A, b, x=None, intercept=True, zeta=1, min_decrease=1e-3, max_iter=1000):
    A = np.array(A)
    b = np.array(b)
    n, k = A.shape
    if x is None:
        x = np.full(k, 1/k)
    if intercept:
        A = A - np.mean(A, axis=0)
        b = b - np.mean(b)
    t = 0
    vals = np.zeros(max_iter)
    eta = n * np.real(zeta ** 2)
    vals_iter = False
    while (t < max_iter) and ((t < 1) or (vals[t - 1] - vals[t] > min_decrease ** 2)):
        x_p = fw_step(A, b, x, eta=eta)
        x = x_p.copy()
        err = np.dot(A, x) - b
        vals[t] = np.real(zeta ** 2) * np.sum(x ** 2) + np.sum(err ** 2) / n
        t += 1

    return {"params": x, "vals": vals}


def collapsed_form(Y, N0, T0):
    N, T = Y.shape
    Y = pd.DataFrame(Y)
    row_mean = Y.iloc[0:N0, T0:T].mean(axis=1)
    col_mean = Y.iloc[N0:N, 0:T0].mean(axis=0)
    overall_mean = Y.iloc[N0:N, T0:T].mean().values[0]
    result_top = pd.concat([Y.iloc[0:N0, 0:T0], row_mean], axis=1)
    result_bottom = pd.concat([col_mean.T, pd.Series(overall_mean)], axis=0)
    return pd.concat([result_top, pd.DataFrame(result_bottom).T], axis=0)


def sc_weight_covariates(
    Y, X_covariates, lambda_est = None, omega_est = None, beta_est = None,
    zeta_lambda=0, zeta_omega=0,
    lambda_intercept=True, omega_intercept=True, 
    max_iter=1000, min_decrease=1e-3,
    update_lambda=True, update_omega=True):


    N, T = Y.shape
    N0, T0 = N - 1, T - 1

    if lambda_est is None:
        lambda_est = [1 / T0] * T0
    if omega_est is None:
        omega_est = [1 / N0] * N0
    if beta_est is None:
        beta_est = np.zeros(len(X_covariates))
    
    def update_weights(Y, lambda_estimation, omega_estimation):
        if lambda_intercept:
            Y_lambda = Y[:N0, :] - np.mean(Y[:N0, :], axis=0)
        else:
            Y_lambda = Y[:N0, :]
            
        if omega_intercept:
            Y_omega = Y[:, :T0].T - np.mean(Y[:, :T0].T, axis = 0)   
        else:
            Y_omega = Y[:, :T0].T
        
        if update_lambda:
            lambda_ = fw_step(Y_lambda[:, 0:T0], Y_lambda[:, T0], lambda_estimation, eta=N0 * zeta_lambda ** 2)
        if update_omega:
            omega_ = fw_step(Y_omega[:, 0:N0], Y_omega[:, N0], omega_estimation, eta=T0 * zeta_omega ** 2)
        
        err_lambda = Y_lambda @ np.concatenate((lambda_, [-1]))
        err_omega = Y_omega @ np.concatenate((omega_, [-1]))
        
        val = zeta_omega ** 2 * np.sum(omega_ ** 2) + zeta_lambda ** 2 * np.sum(lambda_ ** 2) + np.sum(err_omega ** 2) / T0 + np.sum(err_lambda ** 2) / N0
        
        return {
            "val": val,
            "lambda": lambda_,
            "omega": omega_,
            "err_lambda": err_lambda,
            "err_omega": err_omega
        }
    
    # max_iter_1 = max_iter + 1
    # vals = np.zeros(max_iter)
    vals = np.array([])
    t = 0
    y_beta = Y - np.sum(np.multiply(X_covariates, beta_est[:, np.newaxis, np.newaxis]), axis = 0)
    weights = update_weights(y_beta, lambda_est, omega_est)
    vals2 = True
    
    while (t < max_iter) and ((t < 2) or vals2):

        t = t + 1
        coef_grad_beta = []

        for i, x_cov in enumerate(X_covariates):
            
            s_lambda = np.dot(weights["err_lambda"], x_cov[:N0, :]) @ np.concatenate([weights["lambda"], [-1]]) / N0
            s_omega = np.dot(weights["err_omega"], x_cov[:, :T0].T) @ np.concatenate([weights["omega"], [-1]]) / T0

            coef_grad_beta.append(s_lambda + s_omega)
        grad_beta = np.array(coef_grad_beta[0]) * -1
        alpha = 1 / (t + 1)
        beta_est = beta_est - alpha * grad_beta
        y_beta = Y - np.sum(np.multiply(X_covariates, beta_est[:, np.newaxis, np.newaxis]), axis = 0)
        weights = update_weights(y_beta, weights["lambda"], weights["omega"])
        # print(vals, weights["val"])
        vals = np.append(vals, weights["val"])
        # vals[t] = weights["val"]
        tn = len(vals)

        if tn >= 2:
            vals2 = vals[tn - 2] - vals[tn - 1] > min_decrease **2

    return {
        "lambda": weights["lambda"],
        "omega": weights["omega"],
        "beta": beta_est,
        "vals": vals
    }

