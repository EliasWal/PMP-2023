import pandas as pd
import pymc as pm
import numpy as np
from scipy.stats import poisson

data = pd.read_csv('trafic.csv')
trafic_file = data['nr. masini'].values
#
with pm.Model() as model:
	lambda_p = pm.Gamma('lambda_p', alpha=2, beta=1)
	
	trafic_distr = pm.Poisson('trafic_distr', mu=lambda_p, observed=trafic_file)
	
	delta_1 = pm.Normal('delta_1', mu=0, sigma=1)
	delta_2 = pm.Normal('delta_2', mu=0, sigma=1)
	delta_3 = pm.Normal('delta_3', mu=0, sigma=1)
	delta_4 = pm.Normal('delta_4', mu=0, sigma=1)
	
	trafic_ora_7 = trafic_file * pm.math.exp(delta_1)
	trafic_ora_8 = trafic_file * pm.math.exp(delta_2)
	trafic_ora_16 = trafic_file * pm.math.exp(delta_3)
	trafic_ora_19 = trafic_file * pm.math.exp(delta_4)

lambda_hat = np.mean(trafic_file)
alpha = 0.05
n = len(trafic_file)

ore_modificare = [7, 8, 16, 19]

for ora in ore_modificare:
	index_ora = ora - 4
	trafic_ora = trafic_file[index_ora]
	
	interval = poisson.interval(1 - alpha, trafic_ora, loc=0)
	lambda_mode = trafic_ora
	
	print(f"Intervalul pentru λ în timpul orei {ora} este: ({interval[0]}, {interval[1]})")
	print(f"Cea mai probabilă valoare a lui λ în timpul orei {ora} este: {lambda_mode}\n")
