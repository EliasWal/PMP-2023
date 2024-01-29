import numpy as np

N = 10000
k = 30

#a)
# Generare perechi de numere aleatoare X si Y
x, y = np.random.uniform(-1, 1, size=(2, N))

condition_satisfied = x > y**2

# Calculul probabilitatii cerute
probability = condition_satisfied.sum() / N

print("Probabilitatea P(X > Y^2) aproximată folosind metoda Monte Carlo:", probability)


# b)
estimations = []

# Calcularea estimarilor pentru fiecare aproximare
for _ in range(k):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    condition_satisfied = x > y**2
    probability = condition_satisfied.sum() / N
    estimations.append(probability)

# Calcularea mediei si deviatiei standard a estimarilor
mean_estimate = np.mean(estimations)
std_deviation = np.std(estimations)

print("Media:", mean_estimate)
print("Deviatia standard:", std_deviation)
