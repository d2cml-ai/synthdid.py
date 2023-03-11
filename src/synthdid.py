
import numpy as np, pandas as pd


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

v = np.random.rand(10)


def lambda_min(A: np.ndarray, b, x, eta, zeta, maxIter: int, minDecrease):
    row, col = A.shape
    vals = np.zeros(maxIter)
    t = 0
    dd = 1

    while (t < maxIter and (t < 2 or dd > minDecrease)):
        t+=1
        Ax = A.dot(x)
        hg = np.transpose(Ax - b).dot(A) + eta * x
        i = np.argmin(hg)
        dx = -x.copy()

        dx[i] = 1 - x[i]
        v = np.abs(np.min(dx)) + np.abs(np.max(dx))
        
        if v == 0:
            x = x
        else: 
            derr = A[:, i] - Ax
            step = -np.transpose(hg).dot(dx) / (np.sum(derr**2) + eta * np.sum(dx**2))
            conststep = min([1, max([0, step])])
            x = x + conststep * dx
        if t > 1:
            dd = abs(np.sum(vals[t-1] - vals[t-2]))

        vals[t-1] = np.sum(np.abs(Ax - b)) + eta * np.sum(np.abs(x))

    return x


quota = pd.read_stata("data/quota_example.dta", convert_categoricals=False)

unit = "country"
time = "year"

treatment = "quota"
outcome = "womparl"

data_cols = [unit, time, treatment, outcome]
data_0 = quota[data_cols]
data_0[treatment] = data_0[treatment].values.astype(int)

data_1 = (
    data_0
    .groupby(unit, group_keys=False).apply(lambda x: x.assign(
        treated=x.quota.max(),
        ty=np.where(x.quota == 1, x.year, np.nan),
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

# NT = data_1[unit].value_counts().values.min()

N = len(np.array(np.unique(data_1[unit])))

by_years = np.unique(data_1["tyear"])
by_years = by_years[~np.isnan(by_years)]

num_col = data_1.select_dtypes(np.number).columns
data_ref = data_1[num_col].fillna(0)
data_ref[unit] = data_1[unit]
data_ref
################### self.data_ref, self_by_years

### Estimate tau 
mt = 1

tau, tau_wt = np.zeros(len(by_years)), np.zeros(len(by_years))
w_omega, w_lambda = np.ones(len(by_years)), np.ones(len(by_years))

# for
this_year = by_years[len(by_years) - 1]

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
Npre = len(np.unique(data_ref.query(f"{time} < {this_year}")[time]))
Npost = yNT - Npre
ndiff = yNg - Npre
ydiff = yData[outcome].diff()
first = np.arange(yN) % yNT == 0

postt = yData[time] >= this_year
dropc = first + postt + yData["treated"]

preDiff = ydiff[dropc == 0]
sig_t = np.sqrt(np.var(preDiff))
EtaLambda = 1e-6
EtaOmega = (yNtr * yTpost) ** (1 / 4) if mt != 3 else 0

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

A_o = (A[:, :Npre].T - np.mean(A[:, :Npre], axis=1)).T
# b_ = 
b_ = Y1[outcome]
b_o = b_[:Npre] - np.mean(b_)
len(b_o)

## lambda

lambda_l = np.repeat([1 / A_l.shape[1]], A_l.shape[1])
lambda_o = np.repeat([1 / A_o.shape[0]], A_o.shape[0])

mindec = (1e-5 * sig_t) ** 2
eta_o = Npre * yZetaOmega ** 2
eta_l = yNco * yZetaLambda ** 2


lambda_l = lambda_min(A_l, b_l, lambda_l, eta_l, yZetaLambda, 100, mindec)
lambda_l1 = sparsify(lambda_l)
lambda_l2 = lambda_min(A_l, b_l, lambda_l1, eta_l, yZetaLambda, 1000, mindec)


lambda_o = lambda_min(A_o, b_o, lambda_o, eta_o, yZetaOmega, 100, mindec)
lambda_o1 = sparsify(lambda_o)
lambda_o2 = lambda_min(A_o, b_o, lambda_o1, eta_o, yZetaOmega, 1000, mindec)

ytra = Y.to_numpy()
l_o = (np.concatenate([-lambda_o2, np.repeat([1 / yNtr], yNtr)]))
l_o @ ytra[1]


Y.iloc[:, 1]
ytra[: , 1]

# ( @ ytra) @ (np.concatenate([-lambda_l2 ,
# np.repeat([1 / Npost], Npost)]
# ))

lambda_o2


np.concatenate([[-lambda_o], np.repeat([1/yNtr], yNtr)]) @ ytra


tau[1] = np.transpose(np.array([-lambda_o, np.repeat(1/yNtr, yNtr)])) * ytra #@ np.array([-lambda_l, np.repeat(1/Npost, Npost)])
tau_wt[1] = yNtr * Npost




for i in range(len(by_years)):
    this_year = by_years[i]
    



class sdid:
    def __init__(self, data):
        self.data = data
        self.Y = None
        self.w = None
        self.N0 = None
        self.T0 = None
        self.summary = None

    def panel_data(self):
        self.data
        # return a vector of N0 and T0
        # return Y
    def setup(self):
        self.Y
        # return a data for each year
    def s_did(self):
        self.Y
        self.N0
        self.T0
    def summary(self):
        print(self.summary)
    