import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

def generate_data(order=2, num_points=100):
    # Load dummy data
    dummy_data = np.loadtxt('../dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    # Polynomial features
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    return x_1s, y_1s

def plot_scatter(x, y):
    plt.scatter(x[0], y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot')
    plt.show()

def inference_and_plot(order, beta_sd, x_1s, y_1s, num_samples=2000, random_seed=42):
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=beta_sd, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(num_samples, return_inferencedata=True, random_seed=random_seed)

    az.plot_posterior(idata_p, var_names=['α', 'β', 'ε'], hdi_prob=0.95)
    plt.title(f'Posterior with Beta Distribution (sd={beta_sd})')
    plt.show()

def main():
    # Step 1: Change Polynomial Order to 5
    order = 5
    x_1s, y_1s = generate_data(order)
    plot_scatter(x_1s, y_1s)

    # Step 2a: Inference with Beta Distribution (sd=100)
    inference_and_plot(order, beta_sd=100, x_1s=x_1s, y_1s=y_1s)

    # Step 2b: Inference with Beta Distribution (sd=np.array([10, 0.1, 0.1, 0.1, 0.1]))
    inference_and_plot(order, beta_sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), x_1s=x_1s, y_1s=y_1s)

    # Step 3: Increase the Number of Data Points to 500
    num_points = 500
    x_1s, y_1s = generate_data(order, num_points)
    plot_scatter(x_1s, y_1s)

    # Step 4: Cubic Model (order=3), WAIC, LOO, and Comparison
    order = 3
    x_1s, y_1s = generate_data(order)

    with pm.Model() as model_cubic:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_cubic = pm.sample(2000, return_inferencedata=True, random_seed=42)

    # Calculate WAIC and LOO
    waic_cubic = az.waic(idata_cubic)
    loo_cubic = az.loo(idata_cubic)

    # Plot WAIC and LOO comparison
    az.plot_compare({'Cubic': waic_cubic}, ic='waic', scale='deviance')
    plt.title('WAIC Comparison: Cubic Model')
    plt.show()

    az.plot_compare({'Cubic': loo_cubic}, ic='loo', scale='deviance')
    plt.title('LOO Comparison: Cubic Model')
    plt.show()

if __name__ == "__main__":
    main()
