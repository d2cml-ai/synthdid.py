import pandas as pd
import numpy as np

from src import solver

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

def synthdid_estimate(
    df, 
    pre_term, 
    post_term, 
    treatment
):
    
    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)
    n_treat = len(treatment)
    n_post_term = len(Y_post_t)

    zeta = solver.est_zeta(n_treat, n_post_term, Y_pre_c)
    hat_omega = solver.est_omega(solver.l2_loss, Y_pre_c, Y_pre_t, zeta)
    hat_lambda = solver.est_lambda(solver.l2_loss, Y_pre_c, Y_post_c)

    result = pd.DataFrame({"actual_y": target_y(df, pre_term, post_term, treatment)})
    actual_post_treat = result.loc[post_term[0]: , "actual_y"].mean()

    result["sdid"] = sdid_trajectory(hat_omega, hat_lambda, Y_pre_c, Y_post_c)

    pre_sdid = result["sdid"].head(len(hat_lambda)) @ hat_lambda
    post_sdid = result.loc[post_term[0]: , "sdid"].mean()

    pre_treat = (Y_pre_t.T @ hat_lambda).values[0]
    counterfactual_post_treat = pre_treat + (post_sdid - pre_sdid)

    return actual_post_treat - counterfactual_post_treat

def estimated_params(df, pre_term, post_term, treatment):

    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = gen_data(df, pre_term, post_term, treatment)
    n_treat = len(treatment)
    n_post_term = len(Y_post_t)

    zeta = solver.est_zeta(n_treat, n_post_term, Y_pre_c)
    hat_omega = solver.est_omega(solver.l2_loss, Y_pre_c, Y_pre_t, zeta)
    hat_lambda = solver.est_lambda(solver.l2_loss, Y_pre_c, Y_post_c)

    return {"zeta": zeta, "hat_omega": hat_omega, "hat_lambda": hat_lambda}