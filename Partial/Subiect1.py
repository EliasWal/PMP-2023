import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#1)
np.random.seed(42)

def aruncare_moneda(n, prob_stema):
    return np.sum(np.random.uniform(0, 1, n) < prob_stema)

def simulare_joc(num_simulare=20000):
    castiguri_p0 = 0 # Numărul de jocuri câștigate de p0
    castiguri_p1 = 0 # Numărul de jocuri câștigate de p1

    for _ in range(num_simulare):
        incepe_p0 = np.random.choice([True, False])  # Decide cine începe
        if incepe_p0:
            steme_p0 = aruncare_moneda(1, 1/2)  # Moneda necinstita a lui p0
            steme_p1 = aruncare_moneda(steme_p0 + 1, 2/3)  # Moneda cinstita a lui p1
        else:
            steme_p1 = aruncare_moneda(1, 2/3)  # Moneda cinstita a lui p1
            steme_p0 = aruncare_moneda(steme_p1 + 1, 1/2)  # Moneda necinstita a lui p0
        
        if steme_p0 >= steme_p1:
            castiguri_p0 += 1
        else:
            castiguri_p1 += 1

    prob_castig_p0 = castiguri_p0 / num_simulare
    prob_castig_p1 = castiguri_p1 / num_simulare

    print(f"Probabilitatea de câștig pentru p0: {prob_castig_p0}")
    print(f"Probabilitatea de câștig pentru p1: {prob_castig_p1}")


simulare_joc()

#2)
#Crearea modelului Bayesian
model = BayesianNetwork([('IncepeP0', 'StemeP0'), ('IncepeP1', 'StemeP1'), ('StemeP0', 'CastigP0'), ('StemeP1', 'CastigP1')])

# Estimarea parametrilor cu TabularCPD
cpd_incepe_p0 = TabularCPD(variable='IncepeP0', variable_card=2, values=[[0.5], [0.5]])
cpd_incepe_p1 = TabularCPD(variable='IncepeP1', variable_card=2, values=[[0.5] , [0.5]])

cpd_steme_p0 = TabularCPD(variable='StemeP0', variable_card=2,
                         evidence=['IncepeP0'], evidence_card=[2],
                         values=[[1/2, 1/3], [1/2, 2/3]])

cpd_steme_p1 = TabularCPD(variable='StemeP1', variable_card=2,
                         evidence=['IncepeP1'], evidence_card=[2],
                         values=[[2/3, 1/2], [1/3, 1/2]])

cpd_castig_p0 = TabularCPD(variable='CastigP0', variable_card=2,
                          evidence=['StemeP0'], evidence_card=[2],
                          values=[[1, 0], [0, 1]])

cpd_castig_p1 = TabularCPD(variable='CastigP1', variable_card=2,
                          evidence=['StemeP1'], evidence_card=[2],
                          values=[[1, 0], [0, 1]])

# Adăugarea CPD-urilor la model
model.add_cpds(cpd_incepe_p0, cpd_incepe_p1, cpd_steme_p0, cpd_steme_p1, cpd_castig_p0, cpd_castig_p1)

# Calcularea probabilităților marginale cu VariableElimination
inferenta = VariableElimination(model)

# Inferența pentru variabila 'IncepeP0' știind evidența 'StemeP1' = 1
prob_incepe_p0 = inferenta.query(variables=['IncepeP0'], evidence={'StemeP1': 1})
print("Probabilitatea ca jocul să înceapă cu p0:", prob_incepe_p0.values[1])

# Inferența pentru variabila 'IncepeP1' știind 'StemeP1' = 1
prob_incepe_p1 = inferenta.query(variables=['IncepeP1'], evidence={'StemeP1': 1})
print("Probabilitatea ca jocul să înceapă cu p1:", prob_incepe_p1.values[1])

#c)

# Inferența pentru variabila 'StemeP0' știind 'StemeP1' = 0
prob_steme_p0_round1 = inferenta.query(variables=['StemeP0'], evidence={'StemeP1': 0})
print("Probabilitatea de a obține o stemă în prima rundă, dat fiind că în a doua rundă nu s-a obținut nicio stemă:", prob_steme_p0_round1.values[1])
if(prob_steme_p0_round1.values[1] > 0.5):
    print("Stema are mai multe șanse să apară!")
else:
    print("Pajura are mai multe șanse să apară!")
