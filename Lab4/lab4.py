import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import expon


#POISSON

lambda_p = 20

poisson_dist = poisson(mu=lambda_p)

x = np.arange(0, 41)
pmf_values = poisson_dist.pmf(x)
plt.bar(x, pmf_values, label='Poisson(Lambda=20)')
plt.xlabel('Număr de clienți')
plt.ylabel('Probabilitate')
plt.title('Distribuție Poisson')
plt.legend()
plt.show()

prob_10_clients = poisson_dist.pmf(10)

print(f'Probabilitatea de a avea exact 10 clienți într-o oră: {prob_10_clients}')

#NORMAL
mu = 2
sigma = 0.5
norm_dist = norm(loc=mu, scale=sigma)

prob_less_than_1_5 = norm_dist.cdf(1.5)
print(f'Probabilitatea ca timpul să fie mai mic de 1.5 minute: {prob_less_than_1_5}')

x = np.linspace(0, 5, 100)
pdf_values = norm_dist.pdf(x)
plt.plot(x, pdf_values, label='N(2, 0.5)')
plt.xlabel('Timp (minute)')
plt.ylabel('Densitate de probabilitate')
plt.title('Distribuție Normală')
plt.legend()
plt.show()


#EXPONENTIAL
alpha = 3
exp_dist = expon(scale=alpha)

prob_less_than_2 = exp_dist.cdf(2)
print(f'Probabilitatea ca timpul să fie mai mic de 2 minute: {prob_less_than_2}')

x = np.linspace(0, 15, 1000)
pdf_values = exp_dist.pdf(x)
plt.plot(x, pdf_values, label='Exp(3)')
plt.xlabel('Timp (minute)')
plt.ylabel('Densitate de probabilitate')
plt.title('Distribuție Exponentială')
plt.legend()
plt.show()

target_prob = 0.95
max_wait_time = 15

alpha_max = -np.log(1-target_prob)/max_wait_time
print(f'α maxim pentru a servi mancarea în mai puțin de 15 minute tuturor clienților cu o probabilitate de 95%: {alpha_max:.4f}')


# alpha_max = None
# for alpha_candidate in np.linspace(0.001, 10, 1000):
#     cumulative_prob = 1 - np.exp(-alpha_candidate * max_wait_time)
#     if cumulative_prob >= target_prob:
#         alpha_max = alpha_candidate
#         break

# if alpha_max is not None:
#     print(f'α maxim pentru a servi masa în mai puțin de 15 minute tuturor clienților cu o probabilitate de 95%: {alpha_max:.4f}')
# else:
#     print('Nu s-a găsit o soluție.')

# Timpul mediu de așteptare
mean_wait_time = 1 / alpha_max

print(f'Timpul mediu de așteptare pentru a fi servit unui client: {mean_wait_time:.4f} minute')
