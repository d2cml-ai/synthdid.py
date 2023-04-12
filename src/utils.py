import pandas as pd, numpy as np
from numpy.random import default_rng, poisson

def panel_matrices(data: pd.DataFrame(), unit, time, treatment, outcome, covariates = None): #-> data_prep
	if len(np.unique(data[treatment])) != 2:
		print("Error")

	data_ref = pd.DataFrame()
	data_ref[["unit", "time", "outcome"]] = data[[unit, time, outcome]]
	data_ref["treatment"] = data[treatment].to_numpy()
	other = data.drop(columns=[unit, time, outcome, treatment])

	unit, time, outcome, treatment = data_ref.columns

	data_ref = (
		data_ref
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

	break_points = np.unique(data_ref.tyear)
	break_points = break_points[~np.isnan(break_points)]

	units = data_ref[unit]
	num_col = data_ref.select_dtypes(np.number).columns
	data_ref = data_ref[num_col].fillna(0)
	data_ref[unit] = units
	if covariates is not None:
		data_ref = pd.concat([data_ref, other], axis = 1)
		# data_ref[covariates] = data_ref[covariates].fillna(0)s
	data_ref = data_ref.sort_values(["treated", "time", "unit"])

	return (data_ref, break_points)

def collapse_form(Y: np.ndarray, N0: int, T0: int) -> np.ndarray:
	N, T = Y.shape
	Y = pd.DataFrame(Y)
	row_mean = Y.iloc[0:N0, T0:T].mean(axis=1)
	col_mean = Y.iloc[N0:N, 0:T0].mean(axis=0)
	overall_mean = Y.iloc[N0:N, T0:T].mean().values[0]
	result_top = pd.concat([Y.iloc[0:N0, 0:T0], row_mean], axis=1)
	result_bottom = pd.concat([col_mean.T, pd.Series(overall_mean)], axis=0)
	Yc = pd.concat([result_top, pd.DataFrame(result_bottom).T], axis=0)
	return Yc

def sum_normalize(x):
    if np.sum(x) != 0:
        return x / np.sum(x)
    else:
        return np.full(len(x), 1/len(x))
    
def att_mult(Y_beta, omega, _lambda, N1, T1):
    weights_omega = np.concatenate(([-omega], np.full(N1, 1/N1)))
    weights_lambda = np.concatenate(([-_lambda], np.full(T1, 1/T1)))
    return np.dot(weights_omega, Y_beta).dot(weights_lambda)
    
    # omg = np.concatenate(([-omega_est, np.full(N1, 1/N1)]))
			# lmd = np.concatenate(([-lambda_est, np.full(T1, 1/T1)]))