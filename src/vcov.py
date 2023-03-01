from  get_data import *
from  placebo_simulations import *
# import  solver
from solver import *
from  synthdid import *

def vcov(object, method = "placebo", replications = 200):
    Y_pre_c, Y_pre_t, Y_post_c, Y_post_t = object["Y_pre_c"], object["Y_pre_t"], object["Y_post_c"], object["Y_post_t"]
    pre_term, post_term = object["pre_term"], object["post_term"]

    if method == "placebo":

        assert object["n_treat"] < Y_pre_c.shape[1]
        control_names = Y_pre_c.columns

        result_tau_object = []

        for i in tqdm(range(replications)):
            # setup
            np.random.seed(seed=0 + i)
            placebo_t = np.random.choice(control_names, object["n_treat"], replace=False)
            placebo_c = [col for col in control_names if col not in placebo_t]
            pla_Y_pre_t = Y_pre_c[placebo_t]
            pla_Y_post_t = Y_post_c[placebo_t]
            pla_Y_pre_c = Y_pre_c[placebo_c]
            pla_Y_post_c = Y_post_c[placebo_c]

            pla_result = pd.DataFrame(
                {
                    "pla_actual_y": pd.concat([pla_Y_pre_t, pla_Y_post_t]).mean(
                        axis=1
                    )
                }
            )

            post_placebo_treat = pla_result.loc[
                post_term[0] :, "pla_actual_y"
            ].mean()

            # estimation
            ## sdid
            pla_zeta = est_zeta(object["n_treat"], object["n_post_term"], pla_Y_pre_c)

            pla_hat_omega = est_omega(l2_loss, pla_Y_pre_c, pla_Y_pre_t, pla_zeta)
            pla_hat_lambda = est_lambda(l2_loss, pla_Y_pre_c, pla_Y_post_c)

            # prediction
            ## sdid
            pla_hat_omega = pla_hat_omega[:-1]
            pla_Y_c = pd.concat([pla_Y_pre_c, pla_Y_post_c])
            n_features = pla_Y_pre_c.shape[1]
            start_w = np.repeat(1 / n_features, n_features)

            _intercept = (start_w - pla_hat_omega) @ pla_Y_pre_c.T @ pla_hat_lambda

            pla_result["object"] = pla_Y_c.dot(pla_hat_omega) + _intercept

            # cal tau
            ## sdid
            pre_object = pla_result["object"].head(len(pla_hat_lambda)) @ pla_hat_lambda
            post_object = pla_result.loc[post_term[0] :, "object"].mean()

            pre_treat = (pla_Y_pre_t.T @ pla_hat_lambda).values[0]
            object_counterfuctual_post_treat = pre_treat + (post_object - pre_object)

            result_tau_object.append(
                post_placebo_treat - object_counterfuctual_post_treat
            )
    return (np.var(result_tau_object))







