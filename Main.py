# Import all modules used in this script
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from math import sqrt

#Import all functions (models) from Monte_Carlo_GBM.py
from Monte_Carlo_GBM import Monte_Carlo_SDE_GBM
from Monte_Carlo_GBM import Alternative_Monte_Carlo_GBM
from Monte_Carlo_GBM import Analytic_Solution_GBM
from Monte_Carlo_GBM import evaluate_algorithms

#Import all functions (post-proc) from Post_processing.py
from Post_processing import plot_price_with_time
from Post_processing import plot_histogram
from Post_processing import plot_histogram_together
from Post_processing import plot_kde
from Post_processing import plot_kde_together
from Post_processing import print_percentiles
from Post_processing import show_stats
from Post_processing import plot_histograms_figure2
from Post_processing import plot_kdes_figure3


# Import data from Excel as a DataFrame for python; date is parsed and converted to python date format 
ETF_data = pd.read_excel('ETF_data.xlsx', index_col=0, parse_dates = ['date'])

# Date for indexation of dataframe
ETF_data.index.name = 'date'
# Check whether there is any NaN values in the dataset
print('Is there any NaN values in the dataset :', ETF_data.isnull().values.any())
# Look at the values of the dataframe
values = ETF_data.values
# Ensure all data is float
values = values.astype('float32')

# Add column for the return of the ETF
ETF_data['return'] = (ETF_data['close'].pct_change()).shift(-1)
# Add column for the logarithmic return of the ETF
ETF_data['ln_return'] = (np.log(ETF_data['close'].shift(-1) / ETF_data['close']))

# Counting number of days in a year
count_yearly_days = ETF_data['close'].resample("Y").count()
# Removing the last row of the dataset because 2018 is not finished yet
count_yearly_days = count_yearly_days[:-1]
# Averaging the number of days for each year to get the average number of trading days and change it as an integer
trading_days_per_year = int(count_yearly_days.mean())
# Print the chosen number of trading days per year
print('Number of trading days per year :', trading_days_per_year)

# Show the number of days for the whole period of trading
delta_days = (ETF_data.index.date[-1]-ETF_data.index.date[0]).days
print('Number of days between start and end period :', delta_days)
# Calculate cumulative return of investment
cumu_return = (ETF_data.close[-1] - ETF_data.close[0]) / ETF_data.close[0]
# Estimate annualized return of investment
annual_return = (1 + cumu_return)**(365/delta_days) - 1
# Calculate the average daily return
mean_daily_return = ETF_data['return'].mean()

# Calculate the daily standard deviation (volatility) of returns
daily_sigma = np.std(ETF_data['return'])
# Annualized volatility
annual_sigma = daily_sigma * sqrt(trading_days_per_year)


print()
print()
# Printing both properties we are interested in percentage and rounded numbers
print('Annualized return :', str(round(annual_return,5) * 100) + '%')
print('Annualized volatility :', str(round(annual_sigma,4) * 100) + '%')

print('\n')







# Creating a dictionary for the different models used in simulation
models = dict()
models['sde_gbm'] = Monte_Carlo_SDE_GBM
models['analytic_exp_gbm'] = Alternative_Monte_Carlo_GBM
models['analytic'] = Analytic_Solution_GBM

#### Parameters to save figures
# Not saving but showing plot
#save_fig = 0
# Saving plot to an external file (by default)
save_fig = 1


########################################################

# S0 corresponds to the starting price of the stock
# sigma is the daily volatility
# mu correponds to the mean daily returns
# T is the number of years for the simulation
# n_days is the number of days of the simulation
# dt corresponds to the timestep of 1 day
# n_ETF corresponds to the number of ETF held
S0 = ETF_data.close[-1]
sigma = annual_sigma
mu = annual_return
T = 10
dt = 1/trading_days_per_year
n_ETF = 10000 / S0

print('-----Properties for the Monte Carlo simulations:------')
print('Initial price S0: ' + str(round(S0,1)) + '$' )
print('Annualized return mu: ' + str(round(mu,4) * 100) + '%' )
print('Annualized volatility sigma: ' + str(round(sigma,4) * 100) + '%')
print('Period of simulation: ' + str(T) + ' years')
print('Timestep: ' + str(round(dt,3)) + ' days')
print('Number of ETF(s) held: ' + str(round(n_ETF,3)))

print()

num_iterations = 100
print('Number of iterations per simulations: ' + str(num_iterations))
print('\n')

Matrix_results = evaluate_algorithms(models, num_iterations, S0, sigma, mu, T, dt, n_ETF)

# For the plots of price with time, the analytical solution must be avoided and new lists for the model and names are created without it
models_no_analytic =[model for name, model in models.items()]
# The last element (analytic expression) of the list is removed
models_no_analytic = models_no_analytic[:-1]

names_no_analytic=[name for name, model in models.items()]
names_no_analytic = names_no_analytic[:-1]

# The results are looped over all models except the final one (analytic solution) to plot the price against time
for i, name, model in zip(range(len(Matrix_results)-1), names_no_analytic, models_no_analytic):
    plot_price_with_time(str(name), Matrix_results[i], T, dt, save_fig)

# Creating a list of colors to be passed as arguments for the plots
colors = list()
colors.extend(['blue','red','green'])

# Looping over different models and arguments
for i, (name, model), color in zip(range(len(Matrix_results)), models.items(), colors):
    plot_histogram(str(name), Matrix_results[i], color, save_fig) # plotting histograms figure for each model
    plot_kde(name, Matrix_results[i], color,save_fig) # plotting kde figures for each model
    show_stats(name, Matrix_results[i]) # some stats for each model
    print_percentiles(name, Matrix_results[i]) # print the percentiles for each model

# Different plots and subplots
plot_kde_together(models, Matrix_results, colors, save_fig) 
plot_histogram_together(models, Matrix_results, colors, save_fig)
plot_histograms_figure2(models, Matrix_results, colors, save_fig)
plot_kdes_figure3(models, Matrix_results, colors, save_fig)




