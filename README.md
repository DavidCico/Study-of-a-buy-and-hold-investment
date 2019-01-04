# Buy-and-hold investment study
This code shows an exmaple of a buy-and-hold investment of an ETF using python, and Monte Carlo methods to predict the investment return in the future.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need Python 3.x to run the following code.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems. Python needs to be first installed then SciPy, and finally Seaborn as there are dependencies on packages.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

Alongside Python, the SciPy packages are also required. In Ubuntu and Debian, the SciPy ecosystem can be installed by:

    $ sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

Finally, the latest release of Seaborn visualization package, which can be installed with pip:
    
    $ pip install seaborn

For other Linux flavors, OS X and Windows, packages are available at:

http://www.python.org/getit/  
https://www.scipy.org/install.html  
https://seaborn.pydata.org/installing.html#installing


## File descriptions

* 'ETF_data' which is a univariate time series of the price history of the ETF.
* 'Main.py' which contains the main procedure, as well as the data pre-processing of the xlsx file 'ETF_data.xlsx'
* 'Monte_Carlo_GBM.py' which contains the different algorithms used for comparison.
* 'Post_processing.py' where all the functions for post-processing (plots, information, descriptive statistics) are implemented.

### Running the program

The code is ready to be used and just requires running the following command:

    $ python Main.py

The code is well commented and easy to understand. The different parameters calculated and used for the simulations are:
``` python
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

# num_iterations is the number of times the random process is repeated (Monte Carlo simulations)
num_iterations = 100
```

![alt text](https://github.com/DavidCico/Study-of-buy-and-hold-investment/blob/master/Example_Results/analytic_exp_gbm.png = 24x48)

## Contributing

Please read [CONTRIBUTING.md](https://github.com/DavidCico/Study-of-buy-and-hold-investment/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **David Cicoria** - *Initial work* - [DavidCico](https://github.com/DavidCico)

See also the list of [contributors](https://github.com/DavidCico/Study-of-buy-and-hold-investment/graphs/contributors) who participated in this project.
