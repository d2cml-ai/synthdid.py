import src.synthdid as sdid
from src.get_data import fetch_CaliforniaSmoking

df = fetch_CaliforniaSmoking()

PRE_TERM = [1970, 1988]
POST_TERM = [1989, 2000]

tau_hat = sdid.synthdid_estimate(df, PRE_TERM, POST_TERM, ["California"])

est_par = sdid.estimated_params(df, PRE_TERM, POST_TERM, ["California"])
