<!-- - sample_data
  - fetch_CaliforniaSmoking()
- model.py
  - Synthdid().fit()
  - Synthdid().did_potentical_outcomes()
  - Synthdid().sc_potentical_outcomes()
  - Synthdid().sparceReg_potentical_outcome()
  - Synthdid().sdid_trajectory()
  - Synthdid().sdid_potentical_outcome()
  - Synthdid().sparce_sdid_potentical_outcome()
  - Synthdid().estimated_params()
  - Synthdid().hat_tau()
  - Synthdid(Variance).cal_se()
- optimizer.py
  - Optimizer().est_zeta()
  - Optimizer().est_omega()
  - Optimizer().est_lambda()
  - Optimizer().l2_loss
  - Optimizer().rmse_loss
  - Optimizer().rmse_loss_with_V
  - Optimizer().estimate_v
  - Optimizer().est_omega_ADH
  - Optimizer().grid_search_zeta
  - Optimizer().bayes_opt_zeta
  - Optimizer().est_omega_ElasticNet
  - Optimizer().est_omega_Lasso
  - Optimizer().est_omega_Ridge
- plot.py
  - Plot().plot()
  - Plot().comparison_plot()
- summary.py
  - return type "print"
- variance.py
  - estimate_variance("placebo") -->

# Previus Version Comparison

## `pysynthdid.py y synthdid.py`

### Comparing Files

- `model.py` ~== `sdid.py`, `utils.py`
- `optimizer.py` ~== `solver.py`
- `plot.py` ~== `plot.py`
- `variance.py` ~== `vcov.py`

### Differences in Implementation

#### Estimation

In `pysynthdid`, we need to specify the pre- and post-treatment periods with data in pivoted form:

```py
# pysynthdid
from synthdid.model import SynthDID
from synthdid.sample_data import fetch_CaliforniaSmoking
df = fetch_CaliforniaSmoking()

PRE_TEREM = [1970, 1988]
POST_TEREM = [1989, 2000]

TREATMENT = ["California"]

sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
sdid.fit(zeta_type="base")
print("ATT : ", sdid.hat_tau()) # ATT :  -15.607966063072887
```

In `synthdid.py`, we only need to refer to tha target columns, and the pre- and post- treatment will be calculated automatically:

```py
from synthdid.synthdid import Synthdid as sdid
from synthdid.get_data import quota, california_prop99
pd.options.display.float_format = '{:.4f}'.format

sdid = sdid(california_prop99(), unit="State", time="Year", treatment="treated", outcome="PacksPerCapita").fit()
# sdid.summary().summary2

print("ATT: :", sdid.att) # ATT : 	-15.6038
```

The method of estimating the model is the same.

#### Plot

Both cases have the same plot, byt following the code in `Stata` and `R`, the `plot_weights()` method was also implemented.

```py
#pysynthdid
sdid.plot()
```

```py
# synthdid.py
sdid.plot_outcomes() #sdid.plot() # pysynthdid
sdid.plot_weights()
```

#### Internal Funcion

> `optimize.py` - `solver.py`

In `pysynthdid.py`, the function `fmin_slsqp` is used to obtain estimators, compared to `synthdid.py` wich relies on `sc_weights_..`, from the original `R` code and the new implementacion in `Julia`, for difference correction when estimating with covariates.

> There are several methods in `pysynthdid` that are not based on the original paper

- `sdid().fit(zeta_type = ['base', 'grid_search', 'bayesian_opt'])`, where [Other Omega Estimation methods](https://github.com/MasaAsami/pysynthdid/blob/main/notebook/OtherOmegaEstimationMethods.ipynb) concludes that there is no increase in performance.
- In the same file `Significant relaxation of ADH conditions`, it concludes: "The AAtest results below confirm that the original Synthetic Diff. in Diff. has better performance then sdid with Lasso, Rige, and ElasticNet."

> These results were a result of a non-staggered SDID context. It would remain to be verified if this result is shared for staggered SDID contexts.

### Conclusion:

The implementation of `Synthdid.py` and `pysynthdid.py` does not share much in the internal functionaning of the packages, as we start from a different parameter "optimizer". On the other hand, `pysynthdid.py` shows that in not-stagered contexts, Lasso or Ridge models do not show significant iprovement in perfomance compared to the SDID model.
