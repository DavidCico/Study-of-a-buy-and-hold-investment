# Import all modules used in the script
import pandas as pd
import numpy as np
from math import sqrt
from math import exp

# This function implements the Monte Carlo methods from the stochastic differential equation of GBM
def Monte_Carlo_SDE_GBM(S0, sigma, mu, T, dt, n_ETF):
    n_steps = round(T/dt)
    S_price = np.zeros(n_steps) # create an array of n_steps elements
    S_price[0] = n_ETF * S0 # number of ETFs held times price S0 for initialization
    drift = np.zeros(n_steps)
    shock = np.zeros(n_steps)
    for i in range(1, n_steps): # loop for all timesteps
        drift [i] = mu * dt
        shock[i] = sigma * np.random.normal(0,1) * np.sqrt(dt) 
        S_price[i] = S_price[i-1] + S_price[i-1] * (drift[i] + shock[i])
    return S_price

# This function consists of the computation of the analytical solution with timesteps to represent the GBM
def Alternative_Monte_Carlo_GBM(S0, sigma, mu, T, dt, n_ETF):
    n_steps = round(T/dt)
    t = np.linspace(0, T, n_steps)
    Wt = np.random.standard_normal(size = n_steps) 
    Wt = np.cumsum(Wt)*np.sqrt(dt) # standard brownian motion 
    X = (mu-0.5 * sigma**2)*t + sigma*Wt 
    S_price = n_ETF*S0*np.exp(X) # geometric brownian motion 
    return S_price

# This function returns a list values from the direct computation of the analytic solution at a particular time  
def Analytic_Solution_GBM(S0, sigma, mu, T, dt,n_ETF):
    S_prices = list()
    Wt = np.random.normal(loc=0,scale=np.sqrt(T)) # Wt follows a normal distribution with mean = 0, and std = \sqrt(period)
    S_price = n_ETF * S0 * exp((mu - sigma**2 / 2) * T + sigma * Wt )
    S_prices.append(S_price)
    return S_prices

# This function evalutes the different algorithms iteratively and return the results in a list of dataframes
def evaluate_algorithms(models, num_iterations, S0, sigma, mu, T, dt, n_ETF):
    Matrix_results = list()
    for (name, model) in models.items():
        sim_results = pd.DataFrame()
        for i in range(num_iterations):
            price = model(S0, sigma, mu, T, dt, n_ETF)
            sim_results[i] = price
        Matrix_results.append(sim_results)
    return Matrix_results 


# Run a function and return the run time and the result of the function
def calculateRunTime(function, *args):
    startTime = time.time()
    result = function(*args)
    return time.time() - startTime, result


