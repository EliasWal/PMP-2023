import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GRE_SCORE = 550
GPA_SCORE = 3.5

def build_logistic_model(gre_scores, gpa_scores, admission_results):
    with pm.Model() as logistic_model:
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        p = pm.invlogit(beta0 + beta1 * gre_scores + beta2 * gpa_scores)

        admission_likelihood = pm.Bernoulli('admission_likelihood', p=p, observed=admission_results)

    return logistic_model


def plot_decision_boundary(trace, gre_scores, gpa_scores, admission_results):
    beta0_samples = trace.posterior['beta0'].values
    beta1_samples = trace.posterior['beta1'].values
    beta2_samples = trace.posterior['beta2'].values

    p_samples = pm.invlogit(beta0_samples + beta1_samples * gre_scores + beta2_samples * gpa_scores)

    p_mean = np.mean(p_samples, axis=0)

    plt.figure(figsize=(18, 10))
    plt.scatter(gre_scores, gpa_scores, c=admission_results, cmap='viridis', edgecolors='k', marker='o', s=50, alpha=0.8)
    plt.contour(gre_scores, gpa_scores, p_mean.reshape(gre_scores.shape), levels=[0.5], colors='red', linewidths=2)

    hdi = pm.hpd(p_samples)
    plt.fill_between(gre_scores, hdi[:, 0], hdi[:, 1], color='orange', alpha=0.3, label='94% HDI')

    plt.xlabel('GRE Score')
    plt.ylabel('GPA')
    plt.title('Decision Boundary and 94% HDI')
    plt.legend()
    plt.show()





def main():
    # Load data
    data = pd.read_csv('Admission.csv')
    gre_scores = data['GRE'].values
    gpa_scores = data['GPA'].values
    admission_results = data['Admission'].values
    # a)
    # Build logistic model
    logistic_model = build_logistic_model(gre_scores, gpa_scores, admission_results)

    # Sample from the posterior distribution
    with logistic_model:
        trace = pm.sample(10000, tune=4000, chains=4)

    # Plot posterior distributions
    pm.plot_posterior(trace, var_names=['beta0', 'beta1', 'beta2'], figsize=(12, 6))
    plt.show()

    # b)
    plot_decision_boundary(trace, gre_scores, gpa_scores, admission_results)

    # c)
    GRE_SCORE = 550
    GPA_SCORE = 3.5

    p_student_samples = pm.invlogit(trace['beta0'] + trace['beta1'] * GRE_SCORE + trace['beta2'] * GPA_SCORE)

    hdi_prob = pm.hpd(p_student_samples, hdi_prob=0.9)

    print(f"90% HDI for probability of admission: {hdi_prob}")

    # d)
    GRE_SCORE_NEW = 500
    GPA_SCORE_NEW = 3.2

    p_student_samples_new = pm.invlogit(trace['beta0'] + trace['beta1'] * GRE_SCORE_NEW + trace['beta2'] * GPA_SCORE_NEW)

    hdi_prob_new = pm.hpd(p_student_samples_new, hdi_prob=0.9)

    print(f"90% HDI for probability of admission (new student): {hdi_prob_new}")



if __name__ == "__main__":
    main()
