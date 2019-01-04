# Buy-and-hold investment study
This code shows an exmaple of a buy-and-hold investment using python, and Monte Carlo methods to predict the investment return in the future.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need Python 3.4 or later to run mypy.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux flavors, OS X and Windows, packages are available at

http://www.python.org/getit/


## File descriptions

* 'ETF_data.xlsx' which is a univariate time series of the historical price of the ETF investment.
* 'Main.py' which contains the main procedure, as well as the data pre-processing of the xlsx file 'ETF_data.xlsx'
* 'Monte_Carlo_GBM.py' which contains the different algorithms used for comparison.
* 'Post_processing.py' where all the functions for post-processing (plots, information, descriptive statistics) are implemented.

### Running the program

To run the program

    python Main.py



## Contributing

Please read [CONTRIBUTING.md](https://github.com/DavidCico/Study-of-buy-and-hold-investment/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **David Cicoria** - *Initial work* - [DavidCico](https://github.com/DavidCico)

See also the list of [contributors](https://github.com/DavidCico/Study-of-buy-and-hold-investment/graphs/contributors) who participated in this project.
