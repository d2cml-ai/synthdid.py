
import itertools, pandas as pd, numpy as np
from sdid import sdid



def bootstrap_se(data_ref, n_reps = 50):
    uniqueID = np.unique(data_ref.unit)
    N = len(uniqueID)
    def theta_bt():
        sample_id = np.random.choice(uniqueID, replace=True, size=N)
        def sample_concat(_id):
            sample_id_n = sample_id[_id]
            data_c = data_ref.query("unit in @sample_id_n")
            data_c = data_c.assign(unit1=str(data_c['unit']) + str(_id))
            return data_c
        sampled_df = pd.concat([sample_concat(i) for i in range(N)], ignore_index=True)
        if len(np.unique(sampled_df.treatment)) != 2:
            theta_bt()
        att_aux = sdid(sampled_df, "unit1", "time", "treatment", "outcome")["att"]
        return att_aux
    t = 0
    att_bt = np.array([])
    while t < n_reps:
        t+= 1
        aux = theta_bt()
        att_bt = np.append(att_bt, aux)
    se_bootstrap = np.sqrt(1 / n_reps * np.sum((att_bt - np.sum(att_bt / n_reps)) ** 2))
    return se_bootstrap

def placebo_se(data_ref, n_reps=50):
    tr_years = data_ref.query("time == tyear and tyear != 0").time
    N_tr = len(tr_years)
    df_co = data_ref.query("treated == 0")
    units_df_co = np.unique(df_co.unit)
    N_co = len(units_df_co)
    N_aux = N_co - N_tr
    
    def theta_pb():
        plabeo_years = pd.DataFrame({
            "unit": np.random.choice(units_df_co, size=N_tr),
            'tyear1': tr_years
        })
        aux_data = df_co.merge(plabeo_years, on="unit", how='outer').sort_values("tyear1")
        aux_data = aux_data.assign(
            tyear=aux_data.tyear1.fillna(aux_data["tyear"])
        )
        aux_data = aux_data.assign(
            treatment=np.where(((aux_data.tyear != 0) & (aux_data.time == aux_data.tyear)), 1, 0)
        ).groupby("unit", group_keys=False).apply(
            lambda x: x.assign(
                treated=x["treatment"].max()
            )
        ).reset_index(drop=True)
        att = sdid(aux_data, "unit", "time", "treatment", "outcome")
        return att["att"]
    
    t = 0
    att_pb = np.array([])
    while t < n_reps:
        t += 1
        aux = theta_pb()
        att_pb = np.append(att_pb, aux)
    se_placebo = np.sqrt(1 / n_reps * np.sum((att_pb - np.sum(att_pb / n_reps)) ** 2))
    return se_placebo

def jackknife_se(data_ref, time_break, weights, estimate):
    # data_ref, time_break = estimate["data_ref"], estimate["break_points"]
    uniqID = np.unique(data_ref["unit"])
    N = len(uniqID)
    # weigths = estimate["weights"]
    lambda_estimate, omega_estimate = np.array(weigths["lambda"], dtype = object), np.array(weigths["omega"], dtype = object)

    def theta_jk(ind, _id):
        omega_aux = sum_normalize(omega_estimate[_id])
        if ind < len(omega_estimate):
            omega_aux = sum_normalize(np.delete(omega_estimate[_id], ind))
        lambda_aux = lambda_estimate[_id]
        drop_unit, tyear_filter = uniqID[ind], [0, time_break[_id]]
        
        data_aux = data_ref.query("unit not in @drop_unit").query("tyear in @tyear_filter")
        yng = data_aux.groupby('unit').size().shape[0]
        ynt = data_aux.groupby('unit').size().min()
        N1 = int((data_aux['tyear'] == time_break[_id]).sum() / ynt)
        npre = data_aux[data_aux['time'] < time_break[_id]].groupby('time').size().shape[0]
        T1 = int(ynt - npre)
        tau_wt_aux = N1 * T1

        Y = data_aux.pivot_table(index="unit", columns="time", values="outcome", sort=False)
        Y_beta = np.array(Y)
        nt = Y.shape
        N1, T1 = int(nt[0] - len(omega_aux)), int(nt[1] - len(lambda_aux))
        weights_omega = np.concatenate((-omega_aux, np.full(N1, 1/N1)))
        weights_lambda = np.concatenate((-lambda_aux, np.full(T1, 1/T1)))
        tau_aux = np.dot(weights_omega, Y).dot(weights_lambda)
        return tau_aux
    att_table = pd.concat([theta_jk(i, j) for i, j in itertools.product(range(N), range(len(time_break)))], ignore_index=True)
    result = att_table.groupby('unit', group_keys=True)\
    .apply(
        lambda x: x.assign(
            tau_wt=x['tau_wt_aux'] / x['tau_wt_aux'].sum(),
            # att_aux= (x['tau_aux'] * (x['tau_wt_aux'] / x['tau_wt_aux'].sum())).sum()
            att_aux=x['tau_aux'] * (x['tau_wt_aux'] / x['tau_wt_aux'].sum())
            )
        ).reset_index(drop=True)
    att_aux = result.groupby("unit").sum().att_aux.to_numpy()
    se_jackknife = ((N-1)/N) * (N - 1) * varianza(att_aux)
    return se_jackknife

class Variance:
    def vcov(self, method="placebo", n_reps=50):
        data_ref = self.data_ref
        if method=="placebo":
            se = placebo_se(data_ref, n_reps=50)
        elif method=="bootstrap":
            se = bootstrap_se(data_ref, n_reps=50)
        else:
            time_break, weights = self.ttime, self.weights
            se = jackknife_se(data_ref, time_break, weights)
        self.se = se
        return self