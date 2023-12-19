import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

# Generare date din mixtura de 3 distribuții Gaussiene
np.random.seed(42)
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
mix = np.concatenate([np.random.normal(mu, sigma, size) for mu, sigma, size in zip(means, std_devs, n_cluster)])

# Vizualizare KDE pentru datele generate
az.plot_kde(np.array(mix))
plt.show()

# Definire model pentru mixtura de distribuții Gaussiene cu 2 componente
with pm.Model() as model_2:
    w = pm.Dirichlet('w', a=np.ones(clusters))
    mus = pm.Normal('mus', mu=np.linspace(-5, 5, clusters), sigma=5, shape=clusters)
    sigmas = pm.HalfNormal('sigmas', sigma=5, shape=clusters)
    x_obs = pm.NormalMixture('x_obs', w=w, mu=mus, sigma=sigmas, observed=mix)

# Definire model pentru mixtura de distribuții Gaussiene cu 3 componente
with pm.Model() as model_3:
    w = pm.Dirichlet('w', a=np.ones(clusters))
    mus = pm.Normal('mus', mu=np.linspace(-5, 5, clusters), sigma=5, shape=clusters)
    sigmas = pm.HalfNormal('sigmas', sigma=5, shape=clusters)
    x_obs = pm.NormalMixture('x_obs', w=w, mu=mus, sigma=sigmas, observed=mix)

# Definire model pentru mixtura de distribuții Gaussiene cu 4 componente
with pm.Model() as model_4:
    w = pm.Dirichlet('w', a=np.ones(clusters))
    mus = pm.Normal('mus', mu=np.linspace(-5, 5, clusters), sigma=5, shape=clusters)
    sigmas = pm.HalfNormal('sigmas', sigma=5, shape=clusters)
    x_obs = pm.NormalMixture('x_obs', w=w, mu=mus, sigma=sigmas, observed=mix)

# Calibrare modele
with model_2:
    trace_2 = pm.sample(1000, tune=1000, cores=1)

with model_3:
    trace_3 = pm.sample(1000, tune=1000, cores=1)

with model_4:
    trace_4 = pm.sample(1000, tune=1000, cores=1)

# Evaluare modele folosind WAIC și LOO
waic_2 = az.waic(trace_2)
waic_3 = az.waic(trace_3)
waic_4 = az.waic(trace_4)

loo_2 = az.loo(trace_2)
loo_3 = az.loo(trace_3)
loo_4 = az.loo(trace_4)


# Comparare rezultate WAIC și LOO
print("WAIC pentru modelul cu 2 componente:", waic_2.waic)
print("WAIC pentru modelul cu 3 componente:", waic_3.waic)
print("WAIC pentru modelul cu 4 componente:", waic_4.waic)

print("LOO pentru modelul cu 2 componente:", loo_2.loo)
print("LOO pentru modelul cu 3 componente:", loo_3.loo)
print("LOO pentru modelul cu 4 componente:", loo_4.loo)

