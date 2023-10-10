import matplotlib.pyplot as plt
from math import comb

# Definim probabilitățile
prob_stema = 0.3
prob_banal = 1 - prob_stema
n = 10

# Lista cu toate combinațiile posibile
combinari = ['ss', 'sb', 'bs', 'bb']

# Calculăm probabilitățile pentru fiecare combinare
prob = []

for combinare in combinari:
    k = combinare.count('s')
    prob.append(comb(n, k) * (prob_stema**k) * (prob_banal**(n-k)))

# Generăm graficul
plt.bar(combinari, prob)

# Adăugăm titlul și etichetele axelor
plt.title('Distribuția Variabilelor Aleatoare')
plt.xlabel('Rezultat')
plt.ylabel('Probabilitate')

# Afișăm graficul
plt.show()
