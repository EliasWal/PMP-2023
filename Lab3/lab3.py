from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
# C - cutremur
# I - Incendiu
# A - Alarma

model = BayesianNetwork([('C', 'I'),
                       ('I', 'A'),
                       ('C', 'A')
                       ])

cpd_cutremur = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])

cpd_incendiu = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]],
                          evidence=['C'], evidence_card=[2])

cpd_alarma = TabularCPD(variable='A', variable_card=2,
                        values=[[0.9999, 0.98, 0.05, 0.02],
                                [0.0001, 0.02, 0.95, 0.98]],
                        evidence=['I', 'C'], evidence_card=[2, 2])
# print(cpd_alarma)

model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)
model.get_cpds()
model.check_model()
assert model.check_model()

infer = VariableElimination(model)
result2 = infer.query(variables=['C'], evidence={'A': 1})
print("Probabilitatea ca un cutremur să fi avut loc dat fiind declanșată alarma de incendiu este:\n", result2, '\n')
result3 = infer.query(variables=['I'], evidence={'A' : 0})
print("Probabilitatea ca un incendiu să fi avut loc, fără ca alarma de incendiu să se activeze, este:\n", result3)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
