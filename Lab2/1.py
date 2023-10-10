import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az



nr_simulations = 10000
lambda1 = 4
lambda2 = 6

service = np.zeros(nr_simulations)

for i in range(nr_simulations):
    if np.random.rand() < 0.4 :
        service[i] = stats.expon(scale=1/lambda1).rvs()
    else:
        service[i] = stats.expon(scale=1/lambda2).rvs()


mean_service = np.mean(service)
std_dev_service = np.std(service)

print(f"Media timpului de servire: {mean_service:.2f} ore")
print(f"Deviatia standard a timpului de servire: {std_dev_service:.2f} ore")


plt.hist(service, bins=100, density=True, alpha=0.6, color='b')
plt.xlabel('Timpul de Servire (ore)')
plt.ylabel('Densitatea Probabilității')
plt.title('Densitatea Distribuției Timpului de Servire')
plt.show()
