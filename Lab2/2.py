import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

alpha = [4, 4, 5, 5]
lambda_ = [3, 2, 2, 3]

processing_times = [stats.gamma(a, scale=1/l).rvs(10000) for a, l in zip(alpha, lambda_)]

lambda_latency = 4

latency = stats.expon(scale=1/lambda_latency).rvs(10000)

server = np.random.randint(4, size=10000)

total_times = [processing_times[server][i] + latency[i] for i, server in enumerate(server)]

probability = np.mean(np.array(total_times) > 3)

print(f"Probabilitatea ca un client sa astepte mai mult de 3 milisecunde: {probability}")

plt.hist(total_times, bins=30, density=True, alpha=0.6, color='b')
plt.xlabel('Timp Total (milisecunde)')
plt.ylabel('Densitatea Probabilității')
plt.title('Densitatea Distribuției Timpului Total')
plt.show()