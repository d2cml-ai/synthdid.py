from scipy.stats import norm
import pandas as pd
class Summary:
	def summary(self):
		self.summary2 = pd.DataFrame(
			{
				'ATT': [a.att], 'Std. Err.': [a.se],
				't': [a.att / a.se],
				'P>|t|': [2 * (1 - norm.cdf(abs(a.att / a.se)))]
			}
		)
		return self