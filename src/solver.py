import numpy as np, pandas as pd

def fw_step(A, b, x, eta, alpha=None):
    Ax = np.dot(A, x)

    half_grad = np.dot((Ax - b), A) + eta * np.transpose(x)
    i = np.argmin(half_grad)
    if alpha is not None:
        x *= (1 - alpha)
        x[i] += alpha
        return x
    else:
        d_x = -x.copy()
        d_x[i] = 1 - x[i]
        if np.all(d_x == 0):
            return x
        d_err = A.iloc[:, i] - Ax
        step_upper = np.dot(-half_grad,  d_x)
        step_bot = np.sum(d_err ** 2) + eta * np.sum(d_x ** 2)
        step = step_upper / step_bot
        constrained_step = np.min([1, np.max([0, step])])
        return x + constrained_step * d_x


# x -> lambda(r)
def sc_weight_fw(A, b, x=None, intercept=True, zeta=1, min_decrease=1e-3, max_iter=1000):
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
    # while (t < max_iter) and ((t < 2) or vals_iter):
    while (t < max_iter) and ((t < 1) or vals[t - 1] - vals[t] > min_decrease ** 2):
        x_p = fw_step(A, b, x, eta=eta)
        x = x_p
        err = np.dot(A, x) - b
        vals[t] = np.real(zeta ** 2) * np.sum(x ** 2) + np.sum(err ** 2) / n
        t += 1
        # if t >= 2:
            # vals_iter = vals[t - 1] - vals[t] > min_decrease ** 2
    # print(t)
    return {"params": x, "vals": vals[1:]}


def collapsed_form(Y, N0, T0):
    N, T = Y.shape
    Y = pd.DataFrame(Y)
    row_mean = Y.iloc[0:N0, T0:T].mean(axis=1)
    col_mean = Y.iloc[N0:N, 0:T0].mean(axis=0)
    overall_mean = Y.iloc[N0:N, T0:T].mean().values[0]
    result_top = pd.concat([Y.iloc[0:N0, 0:T0], row_mean], axis=1)
    result_bottom = pd.concat([col_mean.T, pd.Series(overall_mean)], axis=0)
    return pd.concat([result_top, pd.DataFrame(result_bottom).T], axis=0)


# data = california_prop99()
# sparsify = sparsify_function

# unit, time, treatment, outcome = "State", "Year", "treated", "PacksPerCapita"
# tdf, ttime = panel_matrices(data, unit, time, treatment, outcome)

# T_total = 0
# break_points = len(ttime)
# tau_hat, tau_hat_wt = np.zeros(break_points), []

# lambda_estimate, omega_estimate = [], []

# for i, time_eval in enumerate(ttime):
#     times = [time_eval, 0]
#     df_y = tdf.query("tyear in @times")
#     N1 = len(np.unique(df_y.query("treated == 1").unit))
#     T1 = int(np.max(tdf.time) - time_eval + 1)
#     T_total += N1 * T1
#     tau_hat_wt.append(N1 * T1) 
#     Y = df_y.pivot_table(index="unit", columns="time", values="outcome", sort = False)
#     N, T = Y.shape
#     N0, T0 = int(N - N1), int(T - T1)
#     Yc = collapsed_form(Y, N0, T0)

#     prediff = Y.iloc[:N0, :T0].apply(lambda x: x.diff(), axis=1).iloc[:, 1:]
#     noise_level = np.sqrt(varianza(np.array(prediff).flatten()))

#     eta_omega = ((N - N0) * (T - T0))**(1 / 4)
#     eta_lambda = 1e-6

#     zeta_omega = eta_omega * noise_level
#     zeta_lambda = eta_lambda * noise_level
#     min_decrease = 1e-5 * noise_level
    
#     Al, bl = Yc.iloc[:N0, :T0], Yc.iloc[:N0, T0]
#     Ao, bo = Yc.T.iloc[:T0, :N0], Yc.T.iloc[:T0, N0]
   
#     lambda_intercept = True
#     omega_intercept = True
#     max_iter_pre_sparsify, max_iter = 100, 10000
# # if covariates is None or cov_method == "projected":
#     lambda_opt = sc_weight_fw(Al, bl, None, intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)
#     omega_opt = sc_weight_fw(Ao, bo, None, intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)


#     if sparsify is not None:
#         lambda_opt = sc_weight_fw(Al, bl, sparsify(lambda_opt["params"]), intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter)
#         omega_opt = sc_weight_fw(Ao, bo, sparsify(omega_opt["params"]), intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter)

#     lambda_est = lambda_opt["params"]
#     omega_est = omega_opt["params"]
    

#     omg = np.concatenate(([-omega_est, np.full(N1, 1/N1)]))
#     lmd = np.concatenate(([-lambda_est, np.full(T1, 1/T1)]))

#     tau_hat[i] = np.dot(omg, Y) @ lmd


# lambda_opt = sc_weight_fw(Al, bl, None, intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)
# omega_opt = sc_weight_fw(Ao, bo, None, intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)
# lambda_est = lambda_opt["params"]
# omega_opt["params"]


# sc_weight_fw(Al, bl, sparsify_function(lambda_opt["params"]), intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=10000)["params"]

# sc_weight_fw(Ao, bo, sparsify_function(omega_opt["params"]), intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter)["params"]
# lambda_est

# sparsify_function(lambda_opt["params"])