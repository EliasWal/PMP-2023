import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_logistic_model(gre_scores, gpa_scores, admission_results):
    with pm.Model() as logistic_model:
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        p = pm.invlogit(beta0 + beta1 * gre_scores + beta2 * gpa_scores)

        admission_likelihood = pm.Bernoulli('admission_likelihood', p=p, observed=admission_results)

    return logistic_model

def main():
    # Load data
    data = pd.read_csv('Admission.csv')
    gre_scores = data['GRE'].values
    gpa_scores = data['GPA'].values
    admission_results = data['Admission'].values

    # Build logistic model
    logistic_model = build_logistic_model(gre_scores, gpa_scores, admission_results)

    # Sample from the posterior distribution
    with logistic_model:
        trace = pm.sample(10000, tune=2000, chains=2)

    # Plot posterior distributions
    pm.plot_posterior(trace, var_names=['beta0', 'beta1', 'beta2'], figsize=(12, 6))
    plt.show()

if __name__ == "__main__":
    main()
