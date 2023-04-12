def bootstrap_se(estimate, n_reps = 50):
    data_ref = estimate["data_ref"]
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

def placebo_se(estimate, n_reps=50):
    data_ref = estimate["data_ref"]
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
