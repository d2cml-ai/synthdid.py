import numpy as np, pandas as pd
from get_data import quota, california_prop99
from utils import panel_matrices, collapse_form
from solver import fw_step, sc_weight_fw


def sparsify_function(v) -> np.array:
	return np.where(v <= np.max(v) / 4, 0, v)

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
	tau_hat, tau_hat_wt = np.zeros(ttime), np.zeros(ttime)


	lambda_estimate, omega_estimate = [], []

	for time_eval in ttime:
		df_y = tdf.query("tyear in @ [time_eval, 0]")
		N1 = len(np.unique(df_y.query("treated == 1").unit))
		T1 = np.max(tdf.time) - time_eval + 1
		T_total += N1 * T1
		T_post = N1 * T1

		# create Y matrix and collapse it
		Y = np.matrix(df_y.pivot(index=unit, columns=time, values=outcome).iloc[:, 1:])
		N, T = Y.shape
		N0, T0 = N - N1, T - T1
		Yc = collapse_form(Y, N0, T0)

		# calculate penalty parameters
		noise_level = np.sqrt(varianza(np.diff(Y[:N0, :T0], axis=2)))
		eta_omega = ((NT[0] - N0) * (NT[1] - T0))**(1 / 4)
		eta_lambda = 1e-6
  
		zeta_omega = eta_omega * noise_level
		zeta_lambda = eta_lambda * noise_level
		min_decrease = 1e-5 * noise_level

		if covariates is None or cov_method == "projected":
			lambda_opt = sc_weight_fw(Yc[0:N0-1, 0:T0], Yc[0:N0-1, T0], None, intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)
			omega_opt = sc_weight_fw(Yc[0:T0, 0:N0].T, Yc[-1, 0:T0], None, intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter_pre_sparsify)
   
			if sparsify is not None:
				lambda_opt = sc_weight_fw(Yc[0:N0, 0:T0], Yc[0:N0, -1], sparsify(lambda_opt["lambda"]), intercept=lambda_intercept, zeta=zeta_lambda, min_decrease=min_decrease, max_iter=max_iter)
				omega_opt = sc_weight_fw(Yc[0:T0, 0:N0].T, Yc[-1, 0:T0], sparsify(omega_opt["lambda"]), intercept=omega_intercept, zeta=zeta_omega, min_decrease=min_decrease, max_iter=max_iter)
    
			lambda_est = lambda_opt["params"]
			omega_est = omega_opt["params"]

		if covariates is not None and cov_method == "optimized":
			for covar in covariates:
				x_temp = matrix
			xc = collapse_form(x, N0, T0)
			weights = sc_weight_fw()
			omega_est, lambda_est = weights["omega"], weights["lambda"] #modificar si es necesario
			
		
		lambda_estimate.append(lambda_est)
		omega_estimate.append(omega_est)
		l_o = (np.concatenate([-omega_est, np.repeat([1 / yNtr], yNtr)]))
		l_l = (np.concatenate([-lambda_est, np.repeat([1 / Npost], Npost)]))
		tau_hat[time] = (l_o.T @ pd.DataFrame(Y)) @ l_l
		tau_hat_wt[time] = T_post
	
	tau_hat_wt = tau_hat_wt / T_total

	att = np.dot(tau_hat, tau_hat_wt)

	att_info = pd.DataFrame(
		{
			time: ttime,
			att_time : tau_hat,
			att_wt : tau_hat_wt
		}
	)

	return att




  

