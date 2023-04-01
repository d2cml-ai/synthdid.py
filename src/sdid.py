import numpy as np, pandas as pd
from get_data import quota, california_prop99
from utils import panel_matrices, collapse_form
from solver import fw_step, sc_weight_fw


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
	tdf, ttime = panel_matrices(data, unit, time, treatment, outcome)
	if (covariates is not None) and (cov_method == "projected"):
		tdf = projected(tdf)
	
	T_total = 0
	break_points = len(ttime)
	tau_hat, tau_hat_wt = np.zeros(break_points), np.zeros(break_points)

	lambda_estimate, omega_estimate = [], []

	for i, time_eval in enumerate(ttime):
		times = [time_eval, 0]
		df_y = tdf.query("tyear in @times")
		N1 = len(np.unique(df_y.query("treated == 1").unit))
		T1 = int(np.max(tdf.time) - time_eval + 1)
		T_total += N1 * T1
		tau_hat_wt[i] = N1 * T1 
		Y = df_y.pivot_table(index="unit", columns="time", values="outcome", sort = False)
		N, T = Y.shape
		N0, T0 = int(N - N1), int(T - T1)
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

		
	# print(tau_hat_wt, T_total)
	tau_hat_wt = tau_hat_wt / T_total

	att = np.dot(tau_hat, tau_hat_wt)

	att_info = pd.DataFrame(
		{
			"time": ttime,
			"att_time" : tau_hat,
			"att_wt" : tau_hat_wt
		}
	)

	return att






  

