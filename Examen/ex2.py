import numpy as np
import matplotlib.pyplot as plt

N = 10000
k = 30

# Lista pentru a stoca estimarile
estimations = []

# condition_satisfied = x > y**2
# probability = condition_satisfied.sum() / N
# print("Probabilitatea P(X > Y^2) aproximată folosind metoda Monte Carlo:", probability)

# Functie pentru a genera variabile aleatoare repartizate geometric
def generate_geometric(theta, size):
    return np.random.geometric(theta, size)


# Calcularea estimărilor pentru fiecare aproximare
for _ in range(k):
    x = generate_geometric(0.3, N)
    y = generate_geometric(0.5, N)
    condition_satisfied = x > y**2
    probability = condition_satisfied.sum() / N
    estimations.append(probability)
    print(f"Estimarea probabilitatii pentru aproximarea {len(estimations)}: {probability}")


# Calcularea mediei și deviației standard a estimărilor
mean_estimate = np.mean(estimations)
std_deviation = np.std(estimations)

print("Media:", mean_estimate)
print("Deviatia:", std_deviation)
