import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# a)
data_centered = az.load_arviz_data("centered_eight")
data_non_centered = az.load_arviz_data("non_centered_eight")

print("Modelul Centrat:")
print("Numărul de lanțuri:", data_centered.posterior.chain.size)
print("Mărimea totală a eșantionului generat:", data_centered.posterior.draw.size)

print("\nModelul Necentrat:")
print("Numărul de lanțuri:", data_non_centered.posterior.chain.size)
print("Mărimea totală a eșantionului generat:", data_non_centered.posterior.draw.size)

az.plot_posterior(data_centered)

az.plot_posterior(data_non_centered)

# b)
parametrii = ["mu", "tau"]
posterior_centered = data_centered.posterior[parametrii]
posterior_non_centered = data_non_centered.posterior[parametrii]

rhat_centered = az.rhat(posterior_centered)
rhat_non_centered = az.rhat(posterior_non_centered)

autocorr_centered_mu = np.apply_along_axis(lambda x: np.mean(np.correlate(x, x, mode='full')[len(x)-1:]), 0, posterior_centered["mu"])
autocorr_non_centered_mu = np.apply_along_axis(lambda x: np.mean(np.correlate(x, x, mode='full')[len(x)-1:]), 0, posterior_non_centered["mu"])

autocorr_centered_tau = np.apply_along_axis(lambda x: np.mean(np.correlate(x, x, mode='full')[len(x)-1:]), 0, posterior_centered["tau"])
autocorr_non_centered_tau = np.apply_along_axis(lambda x: np.mean(np.correlate(x, x, mode='full')[len(x)-1:]), 0, posterior_non_centered["tau"])

print("\nModelul Centrat - Rhat pentru mu:", rhat_centered["mu"].values.mean())
print("Modelul Necentrat - Rhat pentru mu:", rhat_non_centered["mu"].values.mean())

print("\nModelul Centrat - Autocorelatie pentru mu:", autocorr_centered_mu.mean())
print("Modelul Necentrat - Autocorelatie pentru mu:", autocorr_non_centered_mu.mean())

print("\nModelul Centrat - Rhat pentru tau:", rhat_centered["tau"].values.mean())
print("Modelul Necentrat - Rhat pentru tau:", rhat_non_centered["tau"].values.mean())

print("\nModelul Centrat - Autocorelatie pentru tau:", autocorr_centered_tau.mean())
print("Modelul Necentrat - Autocorelatie pentru tau:", autocorr_non_centered_tau.mean())
plt.show()

data_centered = az.load_arviz_data("centered_eight")
data_non_centered = az.load_arviz_data("non_centered_eight")

#c)
# Numara divergentele pentru fiecare model
divergences_centered = data_centered.sample_stats.diverging.sum()
divergences_non_centered = data_non_centered.sample_stats.diverging.sum()

print("Numărul de divergențe în modelul centrat:", divergences_centered)
print("Numărul de divergențe în modelul necentrat:", divergences_non_centered)

az.plot_pair(data_centered, var_names=["mu", "tau"], divergences=True)
az.plot_pair(data_non_centered, var_names=["mu", "tau"], divergences=True)

plt.show()
