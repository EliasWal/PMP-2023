import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
data = pd.read_csv('../Lab08_2/Prices.csv')
data['Premium'] = data['Premium'].map({'yes': 1, 'no': 0})
# print(data['Premium'].head())

def model_without_premium(data):
    # a)
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])
        
        step = pm.NUTS()
        trace = pm.sample(5000, step=step, tune=1000, random_seed=42)
        
        az.plot_trace(trace, var_names=['beta1', 'beta2', 'sigma'])
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        
        # b)
        hdi_beta1 = np.percentile(trace.posterior['beta1'], [2.5, 97.5])
        hdi_beta2 = np.percentile(trace.posterior['beta2'], [2.5, 97.5])
        
        print(f'95% HDI pentru beta1: {hdi_beta1}')
        print(f'95% HDI pentru beta2: {hdi_beta2}')
        
        # c)
        # Da, frecventa procesorului si marimea hard diskului au o influenta asupra pretului de vanzare al PC-urilor, deoarece intervalul de incredere nu contine valoarea 0 pentru niciuna dintre beta1 sau beta2.
        
        # d)
        alpha = np.mean(trace.posterior['alpha'])
        beta1 = np.mean(trace.posterior['beta1'])
        beta2 = np.mean(trace.posterior['beta2'])
        sigma = np.mean(trace.posterior['sigma'])
        
        predicted_price = alpha + beta1 * 33 + beta2 * np.log(540)
        predicted_price = np.random.normal(predicted_price, sigma, size=5000)
        
        hdi_pred = az.hdi(predicted_price, hdi_prob=0.9)
        print(f'90% HDI pentru pretul asteptat: {hdi_pred}')
        
        # e)
        # distribuţia predictivă posterioară
        alpha = np.mean(trace.posterior['alpha'])
        beta1 = np.mean(trace.posterior['beta1'])
        beta2 = np.mean(trace.posterior['beta2'])
        sigma = np.mean(trace.posterior['sigma'])
        
        predicted_price2 = alpha + beta1 * 33 + beta2 * np.log(540)
        predicted_price2 = np.random.normal(predicted_price2, sigma, size=5000)
        
        hdi_prediction = az.hdi(predicted_price2, hdi_prob=0.9)
        print(f'90% HDI pentru pretul asteptat folosind distr. predictiva posterioara: {hdi_prediction}')

# bonus
def model_with_premium(data):
    
    # Pentru a vedea cum influenteaza daca produsul este premium sau nu, ar trebui sa obtinem intervalul de credibilitate pentru beta3 = pm.Normal('beta3', mu=0, sigma=10) si sa il adaugam la model. Daca acest interval il exclude pe 0, inseamna ca produsul premium influenteaza pretul.
    with pm.Model() as model:
        alpha1 = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        beta3 = pm.Normal('beta3', mu=0, sigma=10)  # Adăugare pentru PremiumManufacturer
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        mu = alpha1 + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive']) + beta3 * data['Premium'].values
        
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])
        
        step = pm.NUTS()
        trace = pm.sample(5000, step=step, tune=1000, random_seed=42)
        
        hdi_beta1 = np.percentile(trace.posterior['beta1'], [2.5, 97.5])
        hdi_beta2 = np.percentile(trace.posterior['beta2'], [2.5, 97.5])
        hdi_beta3 = np.percentile(trace.posterior['beta3'], [2.5, 97.5])
        
        print(f'95% HDI pentru beta1: {hdi_beta1}')
        print(f'95% HDI pentru beta2: {hdi_beta2}')
        print(f'95% HDI pentru beta3: {hdi_beta3}')
        
        if hdi_beta3[0] > 0 or hdi_beta3[1] < 0:
            print(
                "Intervalul de credibilitate pentru beta3 nu include 0, ceea ce sugerează o influență semnificativă a variabilei Premium asupra prețului.")
        else:
            print(
                "Intervalul de credibilitate pentru beta3 include 0, ceea ce sugerează că variabila Premium nu are o influență semnificativă asupra prețului.")
            
if __name__ == '__main__':
    trace1 = model_without_premium(data)
    trace2 = model_with_premium(data)