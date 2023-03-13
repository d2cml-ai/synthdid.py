
import numpy as np
import pandas as pd
import time as ttm


def sparsify(V):
    n_V = len(V)
    W = np.zeros(n_V)
    for i in range(n_V):
        if V[i] <= np.max(V) / 4:
            W[i] = 0
        else:
            W[i] = V[i]
    W = W / np.sum(W)
    return W

# def lambda_min(A: np.ndarray, b, x, eta, zeta, maxIter: int, minDecrease):
#     row, col = A.shape
#     vals = np.zeros(maxIter)
#     t = 0
#     dd = 1

#     # print("lambda")
#     while (t < maxIter and (t < 2 or dd > minDecrease)):
#         t += 1
#         Ax = A @ x
#         # hg = np.transpose(Ax - b).dot(A) + eta * x
#         hg = np.dot(np.transpose(np.dot(A, x) - b), A) + eta * x
#         i = np.argmin(hg)
#         if(t % 1000 == 0):
#             print(i)
#         dx = -x.copy()

#         dx[i] = 1 - x[i]
#         v = np.abs(np.min(dx)) + np.abs(np.max(dx))
#         if v == 0:
#             x = x
#         else:
#             derr = A[:, i] - Ax
#             step = -np.transpose(hg).dot(dx) / \
#                 (np.sum(derr**2) + eta * np.sum(dx**2))
#             conststep = min([1, max([0, step])])
#             x = x + conststep * dx
#         # if t > 1:
#         #     dd = abs(np.sum(vals[t-1] - vals[t-2]))

#         # vals[t-1] = np.sum(np.abs(Ax - b)) + eta * np.sum(np.abs(x))

#     return x

def varianza(x):
    n = len(x)
    media = sum(x) / n
    return sum((xi - media) ** 2 for xi in x) / (n - 1)

# import rpy2.robjects as r

# r.r('set.seed(1235)')
# r.r('n <- 100')
# r.r('p <- 20')
# A = np.array(r.r('A <- matrix(rnorm(n * p), n, p)'))
# b = np.array(r.r('b <- rnorm(n)'))
# x_true = np.array(r.r('x_true <- runif(p)'))
# eta = r.r('eta <- 0.1')
# zeta = r.r('zeta <- 0.1')
# maxIter = 10000
# minDecrease = 1e-6

def lambda_min(A, b, lambda_v, eta, zeta, maxIter, minDecrease):
    
    row, col = A.shape
    vals = np.zeros(maxIter)
    t = 0
    dd = 1
    while t < maxIter and ((t < 2) or (dd > minDecrease)):    
        t += 1
        Ax = A @ lambda_v
        hg = np.transpose(Ax - b) @ A + eta * lambda_v 
        i = np.argmin(hg)
        dx = -(lambda_v.copy())
        dx[i] = 1 - lambda_v[i]
        v = np.abs(np.min(dx)) + np.abs(np.max(dx))
        # if t % 20 == 0:
        #     print(v)
        if v == 0:
            lambda_v =lambda_v 
        else:
            derr = A[:, i] - Ax
            step = -np.transpose(hg).dot(dx) / \
                (np.sum(derr**2) + eta * np.sum(dx**2))
            conststep = min([1, max([0, step])])
            lambda_v = lambda_v + conststep * dx
    return lambda_v 






class sdid:
    def __init__(self, data, unit, time, treatment, outcome):
        self.data = data
        self.unit = unit
        self.time = time
        self.treatment = treatment
        self.outcome = outcome
        self.Y = None
        self.break_points = None
        self.data_ref = None
        self.w = None
        self.N0 = None
        self.T0 = None
        self.summary = None
        self.new_columns = "unit", "time", "outcome", "treatment"

    def panel_data(self):
        unit, time, treatment, outcome = self.unit, self.time, self.treatment, self.outcome
        data = self.data
        data_0 = pd.DataFrame()
        data_0[["unit", "time", "outcome"]] = data[[unit, time, outcome]]
        data_0["treatment"] = data[treatment].to_numpy()

        unit, time, outcome, treatment = data_0.columns
        self.unit, self.time, self.outcome, self.treatment = data_0.columns
        data_1 = (
            data_0
            .groupby(unit, group_keys=False).apply(lambda x: x.assign(
                treated=x[treatment].max(),
                ty=np.where(x[treatment] == 1, x[time], np.nan),
            ))
            .reset_index(drop=True)
            .groupby(unit, group_keys=False).apply(
                lambda x: x.assign(
                    tyear=np.where(x.treated == 1, x.ty.min(), np.nan)
                )
            )
            .reset_index(drop=True)
            .sort_values(["treated", unit, time])
        )
        N = len(np.array(np.unique(data_1[unit])))

        break_points = np.unique(data_1["tyear"])
        break_points = break_points[~np.isnan(break_points)]
        self.break_points = break_points

        num_col = data_1.select_dtypes(np.number).columns
        data_ref = data_1[num_col].fillna(0)
        data_ref[unit] = data_1[unit]
        self.data_ref = data_ref
        return self

    def synth_did(self):
        break_points, data_ref = self.break_points, self.data_ref
        unit, time, outcome, treatment = self.new_columns
        tau, tau_wt = np.zeros(len(break_points)), np.zeros(len(break_points))
        sig_t0_ref = np.zeros(len(break_points))
        w_omega, w_lambda = np.ones(
            len(break_points)), np.ones(len(break_points))
        for i in range(len(break_points)):
            this_year = break_points[i]
            cond1 = data_ref["tyear"] == this_year
            cond2 = data_ref["tyear"] == 0
            cond = cond1 + cond2
            yData = data_ref.loc[cond, :]

            yNg = len(np.unique(yData[unit]))
            yN = yData.shape[0]
            yNT = yData.groupby(unit).size().min()

            yNtr = sum(cond1) / yNT
            yNco = sum(cond2) / yNT

            yTpost = np.max(yData[time]) - this_year + 1
            Npre = len(np.unique(data_ref.query(
                f"{time} < {this_year}")[time]))
            Npost = yNT - Npre
            ndiff = yNg - Npre
            ydiff = yData[outcome].diff()
            first = np.arange(yN) % yNT == 0

            postt = yData[time] >= this_year
            dropc = first + postt + yData["treated"]

            preDiff = ydiff[dropc == 0] 
            sig_t = np.sqrt(varianza(preDiff.to_numpy()))
            sig_t0_ref[i] = sig_t
            EtaLambda = 1e-6
            EtaOmega = (yNtr * yTpost) ** (1 / 4) if mt != 3 else 1e-6

            yZetaOmega = EtaOmega * sig_t
            yZetaLambda = EtaLambda * sig_t

            ytreated = (
                yData
                .assign(ytreated=lambda x: x['tyear'] == this_year)
                .groupby(unit)
                .agg(ytreat=('ytreated', 'sum'))
                .reset_index()
                .query('ytreat > 0')
                [unit]
                .tolist()
            )

            Y = yData.pivot_table(index=unit, columns=time, values=outcome)
            Y1 = yData.query(f"{unit} in @ytreated")
            Y0 = Y.query(f"{unit} not in @ytreated")

            promt = Y0.iloc[:, Npre:].mean(axis=1)
            A = Y0.values
            A_l = A[:, :Npre] - A[:, :Npre].mean(axis=0)
            b_l = promt - promt.mean()

            A_o = (A[:, :Npre].T - np.mean(A[:, :Npre], axis=1))
            b_ = Y1[outcome]
            b_o = b_[:Npre] - np.mean(b_)

            lambda_l = np.repeat([1 / A_l.shape[1]], A_l.shape[1])
            lambda_o = np.repeat([1 / A_o.shape[1]], A_o.shape[1])

            mindec = (1e-5 * sig_t) ** 2
            eta_o = Npre * yZetaOmega ** 2
            eta_l = yNco * yZetaLambda ** 2

            lambda_l = lambda_min(A_l, b_l, lambda_l,
                                  eta_l, yZetaLambda, 100, mindec)
            lambda_l1 = sparsify(lambda_l)
            lambda_l2 = lambda_min(A_l, b_l, lambda_l1,
                                   eta_l, yZetaLambda, 1000, mindec)

            lambda_o = lambda_min(A_o, b_o, lambda_o,
                                  eta_o, yZetaOmega, 100, mindec)
            lambda_o1 = sparsify(lambda_o)
            lambda_o2 = lambda_min(A_o, b_o, lambda_o1,
                                   eta_o, yZetaOmega, 1000, mindec)

            ytra = yData[self.outcome].to_numpy().reshape(Y.shape)

            l_o = (np.concatenate([-lambda_o2, np.repeat([1 / yNtr], yNtr)]))
            l_l = (np.concatenate([-lambda_l2, np.repeat([1 / Npost], Npost)]))

            tau[i] = (l_o.T @ pd.DataFrame(ytra)) @ l_l
            tau_wt[i] = yNtr * Npost
        tau_wt = tau_wt / np.sum(tau_wt)
        self.att = tau_wt @ tau
        self.tau_wt = tau_wt
        self.tau = tau
        self.sig_t = sig_t0_ref
        return self

    def summary(self):
        print(self.summary)

# if __name__ == "__main__":
mt = 1
# print("california")
# prop_99 = pd.read_stata("http://www.damianclarke.net/stata/prop99_example.dta")
# df = sdid(prop_99, "state", "year", "treated", "packspercapita").panel_data().synth_did()
# df.sig_t
# print(df.att)
print("quota_basic")

quota = pd.read_stata("data/quota_example.dta", convert_categoricals=False)
unit, time, treatment, outcome = "country", "year", "quota", "womparl"
df = sdid(data=quota, unit=unit, time=time,
        treatment=treatment, outcome=outcome).panel_data()

df_1 = df.synth_did()
print(df_1.att)


