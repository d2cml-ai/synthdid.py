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
    while (t < max_iter) and ((t < 1) or vals[t - 1] - vals[t] > min_decrease ** 2):
        x_p = fw_step(A, b, x, eta=eta)
        x = x_p
        err = np.dot(A, x) - b
        vals[t] = np.real(zeta ** 2) * np.sum(x ** 2) + np.sum(err ** 2) / n
        t += 1

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

