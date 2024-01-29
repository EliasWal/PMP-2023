import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


# a.
titanic_data = pd.read_csv('titanic.csv')

# Gestionare date
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# b. Definirea modelului în PyMC3
with pm.Model() as titanic_model:
	# Definirea variabilelor pentru model
	pclass = pm.Data("pclass", titanic_data['Pclass'].values)
	age = pm.Data("age", titanic_data['Age'].values)
	survived = pm.Data("survived", titanic_data['Survived'].values)
	
	beta1 = pm.Normal("beta1", mu=0, sigma=10)
	beta2 = pm.Normal("beta2", mu=0, sigma=10)
	alpha = pm.Normal("alpha", mu=0, sigma=10)
	
	# Modelul liniar
	p_survive = pm.math.invlogit(alpha + beta1 * pclass + beta2 * age)
	
	# Definirea distribuției a priori a parametrului pentru observațiile Bernoulli
	obs_survived = pm.Bernoulli("obs_survived", p=p_survive, observed=survived)

# c. Variabila care influențează cel mai mult rezultatul este Pclass, deoarece are cel mai mare coeficient în modelul liniar.

# d.
with titanic_model:
	# Esantionare din distribuția posterioară
	trace = pm.sample(1000, tune=1000, cores=1)
	
	# Calculul HDI pentru probabilitatea de a supraviețui
	age_30 = 30
	pclass_2 = 2
	p_survive_posterior = pm.math.invlogit(trace['alpha'] + trace['beta2'] * age_30 + trace['beta1'] * pclass_2)
	az.plot_forest(trace, var_names=['alpha', 'beta1', 'beta2'], hdi_prob=0.9)

# Afisarea rezultatelor
plt.show()

