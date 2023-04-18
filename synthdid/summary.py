from scipy.stats import norm
import pandas as pd
class Summary:
	def summary(self):
		att = self.att
		se = self.se
		if se is not None:
			t = att / se
			p_val = 2 * (1 - norm.cdf(abs(att / se)))
		else:
			se = "-"
			t = "-"
			p_val = "-"
			# se = "-"
		self.summary2 = pd.DataFrame(
			{
				'ATT': [att], 'Std. Err.': [se],
				't': [t],
				'P>|t|': [p_val]
			}
		)
		return self