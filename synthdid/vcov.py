
import itertools, pandas as pd, numpy as np
from .sdid import sdid
from .utils import sum_normalize, varianza



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

def jackknife_iteration(data, time_breaks, weights, unit_index: int) -> np.ndarray:
    weighted_atts = np.array([])
    total_treated_unit_periods = data[data.treatment == 1].shape[0]

    for tyear_index, treatment_year in enumerate(time_breaks):
        tyear_data = data[data.tyear.isin([0, treatment_year])]
        N_treated = pd.unique(data[data.tyear == treatment_year].unit).shape[0]
        tyear_omegas =  - weights["omega"][tyear_index]
        N_control = tyear_omegas.shape[0]
        tyear_omegas = np.concatenate([tyear_omegas, np.array([1/N_treated for _ in range(N_treated)])])
        if unit_index < N_control:
            tyear_omegas = np.delete(tyear_omegas, unit_index)
        tyear_lambdas = - weights["lambda"][tyear_index]
        T_post = pd.unique(tyear_data.time).shape[0] - tyear_lambdas.shape[0]
        tyear_treated_unit_periods = N_treated * T_post
        tyear_lambdas = np.concatenate([tyear_lambdas, np.array([1 / T_post for _ in range(T_post)])])
        data_matrix = tyear_data.pivot_table(values = "outcome", index = "unit", columns = "time", sort = False).to_numpy() #type: ignore
        att = tyear_omegas @ data_matrix @ tyear_lambdas.T
        att_weight = tyear_treated_unit_periods / total_treated_unit_periods
        weighted_atts = np.concatenate([weighted_atts, [att * att_weight]])
    
    jk_iteration_att = weighted_atts.sum()
    return np.array([jk_iteration_att])
        


def jackknife_se(data_ref: pd.DataFrame, time_breaks, att, weights):
    
    for tyear in time_breaks:
        if pd.unique(data_ref[data_ref.tyear == tyear].unit).shape[0] == 1:
            raise ValueError(f"Each adoption year must have more than one treated unit. Year {tyear} does not comply")
    
    unique_units = pd.unique(data_ref.unit.unique())
    jackknife_ates = np.array([])

    for unit_index, unit in enumerate(unique_units):
        iteration_ate = jackknife_iteration(
            data_ref[data_ref.unit != unit],
            time_breaks,
            weights,
            unit_index
        )
        jackknife_ates = np.concatenate([
            jackknife_ates,
            iteration_ate
        ])
    
    total_units = unique_units.shape[0]
    var_jackknife = (total_units - 1) / total_units * ((jackknife_ates - att) ** 2).sum()
    se_jackknife = np.sqrt(var_jackknife)

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
            se = jackknife_se(data_ref, time_break, self.att, weights)
        self.se = se
        return self