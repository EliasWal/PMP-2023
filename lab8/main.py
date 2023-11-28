import pymc as pm
import pandas as pd
import numpy as np

data = pd.read_csv('Prices.csv')

if __name__ == '__main__':

	with pm.Model() as model:
		alpha = pm.Normal('alpha', mu=0, sigma=10)
		beta1 = pm.Normal('beta1', mu=0, sigma=10)
		beta2 = pm.Normal('beta2', mu=0, sigma=10)
		sigma = pm.HalfNormal('sigma', sigma=1)  # Folosim HalfNormal pentru a specifica deviația standard pozitivă
		
		mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
		
		y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])
		
		step = pm.NUTS()
		
		trace = pm.sample(5000, step=step, tune=1000, random_seed=42)
	
	pm.summary(trace).round(2)
	pm.plot_trace(trace)
