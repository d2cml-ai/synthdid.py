import pandas as pd
import numpy as np
from scipy.linalg import svd,  norm
import statsmodels.api as sm
import random

def decompose_Y(Y, rank):
    N = Y.shape[0]
    T = Y.shape[1]
    U, D, Vt = svd(Y, full_matrices = False)

    factor_unit = np.matrix(U[:,0:rank] * np.sqrt(N))
    factor_time = np.matrix(Vt.T[:,0:rank] * np.sqrt(T))

    magnitude = D[0:rank]/np.sqrt(N*T)
    L = factor_unit*np.diag(magnitude)*factor_time.T

    E = Y-L
    F = L.mean(axis=1)*np.repeat(1,T)[:,np.newaxis].T + np.repeat(1,N)[:,np.newaxis] * L.mean(axis=0) - np.mean(L)
    M = L - F
    return {"F":F, "M":M, "E":E, "unit_factors":factor_unit}


def ar2_correlation_matrix(ar_coef,T):
  result = np.repeat(0.0,T)
  result[0] = 1
  result[1] = ar_coef[0]/(1-ar_coef[1])

  for t in range(2,T):
    result[t] =  ar_coef[0]*result[t-1] + ar_coef[1]*result[t-2] 

  index_matrix = np.abs(np.arange(0,T)[:,np.newaxis] - np.arange(0,T)[:,np.newaxis].T)
  cor_matrix = result[index_matrix]
  return cor_matrix


def fit_ar2(E):
    T_full = E.shape[1]
    E_ts = E[:,2:T_full]
    E_lag_1 = E[:,1:(T_full-1)]
    E_lag_2 = E[:,0:(T_full-2)]

    a_1 = np.sum(np.diag(E_lag_1*(E_lag_1).T))
    a_2 = np.sum(np.diag(E_lag_2*(E_lag_2).T))
    a_3 = np.sum(np.diag(E_lag_1*(E_lag_2).T))

    matrix_factor = np.matrix([[a_1, a_3], [a_3, a_2]])

    b_1 = np.sum(np.diag(E_lag_1*(E_ts).T))
    b_2 = np.sum(np.diag(E_lag_2*(E_ts).T))

    ar_coef = np.matmul(np.linalg.inv(matrix_factor), [b_1, b_2])
    return np.array(ar_coef)[0]


def estimate_dgp(Y, assignment_vector, rank):
    N = (Y).shape[0]
    T = (Y).shape[1]
    overall_mean = np.mean(Y)
    overall_sd = np.linalg.norm(Y - overall_mean)/np.sqrt(N*T)
    Y_norm = (Y-overall_mean)/overall_sd

    components = decompose_Y(Y_norm, rank = rank)
    M = components["M"]
    F = components["F"]
    E = components["E"]
    unit_factors = components["unit_factors"]

    ar_coef = np.round(fit_ar2(E),2)
    cor_matrix = ar2_correlation_matrix(ar_coef,T)
    scale_sd = np.linalg.norm(E.T*E/N)/np.linalg.norm(cor_matrix)
    cov_mat = cor_matrix*scale_sd

    unit_factors = sm.add_constant(unit_factors)
    model= sm.GLM(assignment_vector, unit_factors, family=sm.families.Binomial())
    model_fit = model.fit()
    assign_prob = model_fit.predict()

    return {"F":F, "M":M, "Sigma":cov_mat, "pi":assign_prob, "ar_coef":ar_coef}

def randomize_treatment(pi, N, N1):
  assignment_sim = np.random.binomial(n = 1, p = pi, size = N)
  index_as = np.array(np.where(assignment_sim==1))[0]
  index_as = index_as.tolist()

  if np.sum(assignment_sim) > N1:
    index_pert = random.sample(index_as,N1)
    assignment_sim = np.repeat(0,N)
    assignment_sim[index_pert] = 1
  elif np.sum(assignment_sim) == 0:
    index_pert = random.sample(np.arange(1,N),N1)
    assignment_sim = np.repeat(0,N)
    assignment_sim[index_pert] = 1
  return assignment_sim


def simulate_dgp(parameters, N1, T1):
    F = parameters["F"]
    M = parameters["M"]
    Sigma = parameters["Sigma"]
    pi = parameters["pi"]

    N = M.shape[0]
    T = M.shape[1]

    assignment = randomize_treatment(pi,N,N1)
    N1 = np.sum(assignment)
    N0 = N - N1
    T0 = T - T1

    Y = F + M + np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, N)
    return {"Y":Y[np.argsort(assignment).tolist(), :], "N0":N0, "T0":T0}



































