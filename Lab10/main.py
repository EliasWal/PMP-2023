import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def run_linear_regression():
    # Load dummy data
    dummy_data = np.loadtxt('../dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5

    # Polynomial features
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    # PyMC3 model
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True, random_seed=42)

    # Visualize the data
    plt.scatter(x_1s[0], y_1s, label='Observed Data')
    plt.xlabel('x')
    plt.ylabel('y')

    # Extract posterior samples
    posterior_samples = idata_p.posterior

    # Generate predictions from the posterior samples
    x_range = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    x_range_p = np.vstack([x_range**i for i in range(1, order+1)])
    x_range_s = (x_range_p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

    # Plot the predicted curve for each posterior sample
    for i in range(len(posterior_samples['α'])):
        y_pred_sample = posterior_samples['α'][i] + np.dot(posterior_samples['β'][i], x_range_s)
        plt.plot(x_range, y_pred_sample, color='gray', alpha=0.1)

    # Highlight the mean predicted curve
    y_pred_mean = np.mean(posterior_samples['α']) + np.dot(np.mean(posterior_samples['β'], axis=0), x_range_s)
    plt.plot(x_range, y_pred_mean, color='red', label='Mean Prediction')

    # Show the plot
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_linear_regression()
