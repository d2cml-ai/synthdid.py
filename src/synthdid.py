import pandas as pd
import numpy as np

from solver import *

def gen_data(
    df,
    pre_term, 
    post_term, 
    treatment: list
):

    control = [col for col in df.columns if col not in treatment]

    Y_pre_c = df.loc[pre_term[0]: pre_term[1], control]
    Y_pre_t = df.loc[pre_term[0]: pre_term[1], treatment]

    Y_post_c = df.loc[post_term[0]: post_term[1], control]
    Y_post_t = df.loc[post_term[0]: post_term[1], treatment]

    return Y_pre_c, Y_pre_t, Y_post_c, Y_post_t


def target_y(df, pre_term, post_term, treatment):
    return df.loc[pre_term[0]: post_term[1], treatment].mean(axis = 1)

def sdid_trajectory(hat_omega, hat_lambda, Y_pre_c, Y_post_c):

    hat_omega = hat_omega[:-1]
    Y_c = pd.concat([Y_pre_c, Y_post_c])
    n_features = Y_pre_c.shape[1]
    start_w = np.repeat(1 / n_features, n_features)

    _intercept = (start_w - hat_omega) @ Y_pre_c.T @ hat_lambda

    return Y_c.dot(hat_omega) + _intercept

def sc_potentical_outcome(Y_pre_c, Y_post_c, hat_omega_ADH):
    return pd.concat([Y_pre_c, Y_post_c]).dot(hat_omega_ADH)

def synthdid_estimate(
    df, 
    pre_term, 
    post_term, 
    treatment
):
    
    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)
    n_treat = len(treatment)
    n_post_term = len(Y_post_t)

    zeta = est_zeta(n_treat, n_post_term, Y_pre_c)
    hat_omega = est_omega(l2_loss, Y_pre_c, Y_pre_t, zeta)
    hat_lambda = est_lambda(l2_loss, Y_pre_c, Y_post_c)

    result = pd.DataFrame({"actual_y": target_y(df, pre_term, post_term, treatment)})
    actual_post_treat = result.loc[post_term[0]: , "actual_y"].mean()

    result["sdid"] = sdid_trajectory(hat_omega, hat_lambda, Y_pre_c, Y_post_c)

    pre_sdid = result["sdid"].head(len(hat_lambda)) @ hat_lambda
    post_sdid = result.loc[post_term[0]: , "sdid"].mean()

    pre_treat = (Y_pre_t.T @ hat_lambda).values[0]
    counterfactual_post_treat = pre_treat + (post_sdid - pre_sdid)

    return { "estimate": (actual_post_treat - counterfactual_post_treat), "df":df, "pre_term": pre_term, "post_term": post_term, "treatment": treatment,
            "Y_pre_c": Y_pre_c, "Y_pre_t": Y_pre_t, "Y_post_c":Y_post_c, "Y_post_t":Y_post_t,
            "n_treat": n_treat, "n_post_term": n_post_term, "zeta":zeta, "hat_omega":hat_omega, "hat_lambda":hat_lambda}

def sdid_params(df, pre_term, post_term, treatment):

    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)
    n_treat = len(treatment)
    n_post_term = len(Y_post_t)

    zeta = est_zeta(n_treat, n_post_term, Y_pre_c)
    hat_omega = est_omega(l2_loss, Y_pre_c, Y_pre_t, zeta)
    hat_lambda = est_lambda(l2_loss, Y_pre_c, Y_post_c)

    return {"zeta": zeta, "hat_omega": hat_omega, "hat_lambda": hat_lambda}

def sc_estimate(
        df, 
        pre_term, 
        post_term, 
        treatment, 
        additional_X = pd.DataFrame(), 
        additional_y = pd.DataFrame()
):

    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)

    result = pd.DataFrame({"actual_y": target_y(df, pre_term, post_term, treatment)})
    actual_post_treat = result.loc[post_term[0]: , "actual_y"].mean()

    hat_omega_ADH = solver.est_omega_ADH(
        Y_pre_c, 
        Y_pre_t, 
        additional_X = additional_X,
        additional_y = additional_y
    )
        
    result["sc"] = sc_potentical_outcome(Y_pre_c, Y_post_c, hat_omega_ADH)
    post_sc = result.loc[post_term[0]:, "sc"].mean()
    counterfactual_post_treat = post_sc

    return actual_post_treat - counterfactual_post_treat

def sc_params(df, pre_term, post_term, treatment, additional_X = pd.DataFrame(), additional_y = pd.DataFrame()):

    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)

    hat_omega_ADH = solver.est_omega_ADH(
        Y_pre_c, 
        Y_pre_t, 
        additional_X = additional_X,
        additional_y = additional_y
    )

    return {"hat_omega": hat_omega_ADH}

def did_estimate(df, pre_term, post_term, treatment):

    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)

    actual_post_treat = Y_post_t.mean(axis=1).mean() - Y_pre_t.mean(axis=1).mean()
    counterfactual_post_treat = Y_post_c.mean(axis=1).mean() - Y_pre_c.mean(axis=1).mean()

    return actual_post_treat - counterfactual_post_treat