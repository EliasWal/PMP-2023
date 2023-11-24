import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

# Parametrii distributiei normale
u = 10  # medie
sigma = 5  # deviatie standard

# Generarea a 200 de timpi medii de așteptare
timpi_medii_asteptare = np.random.normal(loc=u, scale=sigma, size=200)

# Vizualizare histogramă
plt.hist(timpi_medii_asteptare, bins=20, density=True, alpha=0.7, color='blue')
plt.title('Histograma timpilor medii de așteptare')
plt.xlabel('Timp mediu de așteptare')
plt.ylabel('Frecvență relativă')
plt.show()

# Modelul Bayesian
with pm.Model() as model:
    # Distributia a priori pentru parametrul sigma
    sigma_prior = pm.HalfNormal('sigma', sigma=5)
    u_prior = pm.Normal('u', mu=10, sigma=2)

    # Distributia de verosimilitate pentru timpul mediu de așteptare
    obs = pm.Normal('obs', mu=u_prior, sigma=sigma_prior, observed=timpi_medii_asteptare)

    # Estimare distribuție a posteriori pentru sigma
    trace = pm.sample(2000, tune=1000)

# Vizualizare distribuție a posteriori pentru parametrul u
pm.plot_posterior(trace['u'], kde_plot=True)
plt.title('Distribuția a posteriori pentru parametrul mediu u')
plt.xlabel('Timp mediu de așteptare')
plt.show()
