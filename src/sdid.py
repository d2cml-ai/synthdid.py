import numpy as np, pandas as pd
from get_data import quota, california_prop99
from utils import panel_matrices, collapse_form, projected
from solver import fw_step, sc_weight_fw, sc_weight_covariates, contract3


def sparsify_function(v) -> np.array:
	v = np.where(v <= np.max(v) / 4, 0, v)
	return v / sum(v)

def varianza(x):
	n = len(x)
	media = sum(x) / n
	return sum((xi - media) ** 2 for xi in x) / (n - 1)

# scol => unit, tcol => time, ycol=> outcome, dcol => treatment
# data_ref: treated => tunit
def sdid(data: pd.DataFrame, unit, time, treatment, outcome, covariates=None, 
         cov_method="optimized", noise_level=None, eta_omega=None, eta_lambda=1e-6, zeta_omega=None, zeta_lambda=None, omega_intercept=True, lambda_intercept=True, min_decrease=None, max_iter=10000, sparsify=sparsify_function, max_iter_pre_sparsify=100, lambda_estimate=None, omega_estimate=None
		):
	tdf, ttime = panel_matrices(data, unit, time, treatment, outcome, covariates)
	if (covariates is not None) and (cov_method == "projected"):
		tdf = projected(tdf)
	
	T_total = 0
	break_points = len(ttime)
	tau_hat, tau_hat_wt = np.zeros(break_points), np.zeros(break_points)
	N0s, T0s = [], []
	N1s, T1s = [], []
	beta_covariate = []
	Y_beta, Y_units = [], []

	lambda_estimate, omega_estimate = [], []

	for i, time_eval in enumerate(ttime):
		times = [time_eval, 0]
		df_y = tdf.query("tyear in @times")
		N1 = len(np.unique(df_y.query("treated == 1").unit))
		T1 = int(np.max(tdf.time) - time_eval + 1)
		T_total += N1 * T1
		tau_hat_wt[i] = N1 * T1 
		Y = df_y.pivot_table(index="unit", columns="time", values="outcome", sort = False)
		Y_units.append(Y.index)
		N, T = Y.shape
		N0, T0 = int(N - N1), int(T - T1)
		N0s.append(N0)
		T0s.append(T0)
		N1s.append(N1)
		T1s.append(T1)
		Yc = collapse_form(Y, N0, T0)

		prediff = Y.iloc[:N0, :T0].apply(lambda x: x.diff(), axis=1).iloc[:, 1:]
		noise_level = np.sqrt(varianza(np.array(prediff).flatten()))

		eta_omega = ((N - N0) * (T - T0))**(1 / 4)
		eta_lambda = 1e-6

		zeta_omega = eta_omega * noise_level
		zeta_lambda = eta_lambda * noise_level
		min_decrease = 1e-5 * noise_level
		
		Al, bl = Yc.iloc[:N0, :T0], Yc.iloc[:N0, T0]
		Ao, bo = Yc.T.iloc[:T0, :N0], Yc.T.iloc[:T0, N0]
		if covariates is None or cov_method == "projected":
			lambda_opt = sc_weight_fw(Al, bl, None, intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)
			omega_opt = sc_weight_fw(Ao, bo, None, intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)

			if sparsify is not None:
				lambda_opt = sc_weight_fw(Al, bl, sparsify(lambda_opt["params"]), intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter)
				omega_opt = sc_weight_fw(Ao, bo, sparsify(omega_opt["params"]), intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter)

			lambda_est = lambda_opt["params"]
			omega_est = omega_opt["params"]

			

			omg = np.concatenate(([-omega_est, np.full(N1, 1/N1)]))
			lmd = np.concatenate(([-lambda_est, np.full(T1, 1/T1)]))

			tau_hat[i] = np.dot(omg, Y) @ lmd
			Y_beta.append(Y)
		
		if covariates is not None and cov_method == "optimized":
			Yc = np.array(Yc)
			X, Xc = [], []
			for j, cov in enumerate(covariates):
				X_i = df_y.pivot_table(index="unit", columns="time", values=cov, sort = False, fill_value=0)
				X_temp = collapse_form(X_i, N0, T0)
				Xc.append(np.array(X_temp))
				X.append(X_i)
			X = np.array(X)
			Xc = np.array(Xc)
			weights = sc_weight_covariates(
      			Yc, Xc, zeta_lambda = zeta_lambda, zeta_omega = zeta_omega, lambda_intercept = lambda_intercept,
				omega_intercept = omega_intercept, min_decrease = min_decrease, max_iter = max_iter
          		)
			lambda_est = weights["lmbda"]
			omega_est = weights["omega"]
			beta_est = weights["beta"]
			beta_covariate.append(beta_est[0])

			omg = np.atleast_2d(np.concatenate(([-omega_est, np.full(N1, 1/N1)])))
			lmd = np.atleast_2d(np.concatenate(([-lambda_est, np.full(T1, 1/T1)])))

			X_beta = contract3(X, beta_est)
			y_beta = Y - X_beta
			Y_beta.append(y_beta)
			tau_hat[i] = omg.T @ y_beta @ lmd
		

		lambda_estimate.append(lambda_est)
		omega_estimate.append(omega_est)
		weights = {"lambda":lambda_estimate, "omega": omega_estimate}

	tau_hat_wt = tau_hat_wt / T_total

	att = round(np.dot(tau_hat, tau_hat_wt), 5) 

	att_info = pd.DataFrame(
		{
			"time": ttime,
			"att_time" : tau_hat,
			"att_wt" : tau_hat_wt,
			"N0": N0s, "T0": T0s, "N1": N1s, "T1": T1s,
			# "beta_covariate": beta_covariate
		}
	)
	result = {
		"att": att,
		"att_info": att_info,
		"weights": weights,
		"data_ref": tdf, "break_points": ttime,
		"Y_beta": Y_beta, "Y_units": Y_units
	}

	return result






  

