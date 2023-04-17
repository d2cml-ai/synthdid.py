import src.synthdid as sdid
from src.get_data import fetch_CaliforniaSmoking

df = fetch_CaliforniaSmoking()

PRE_TERM = [1970, 1988]
POST_TERM = [1989, 2000]

tau_hat_sdid = sdid.synthdid_estimate(df, PRE_TERM, POST_TERM, ["California"])

tau_hat_sc = sdid.sc_estimate(df, PRE_TERM, POST_TERM, ["California"])

sc_params = sdid.sc_params(df, PRE_TERM, POST_TERM, ["California"])

tau_hat_did = sdid.did_estimate(df, PRE_TERM, POST_TERM, ["California"])