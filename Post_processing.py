# Import all modules used in the script
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns 
import numpy as np

# Subclass the matplotlib.ticker.ScalarFormatter for the order of magnitude of the scientific writing in some plots
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


# This function plots the evolution of the price with time for the different simulations
def plot_price_with_time(name, results_simulation, T, dt, save_fig = 1):
    values = results_simulation.values
    mean_final_price = values[-1].mean() # compute the mean of the last row of results dataframe

    n_steps = round(T / dt)
    t = np.linspace(0, T, n_steps) # creates a time vector to plot from 0 to T with timesteps of n_steps
    fig = plt.figure()
    plt.style.use('bmh')
    title = "Monte Carlo Simulation : " + str(T) + " Years"
    plt.plot(t, results_simulation) # plot the final prices results in function of time
    fig.suptitle(title,fontsize=18, fontweight='bold')
    fig.text(0.15, 0.75, 'Model ' + str(name) , color = 'r' , weight = 'bold', va='center', size = 18, )
    plt.xlabel('Years')
    plt.ylabel('Investment Value ($)')
    plt.grid(True,color='grey')
    plt.axhline(y=mean_final_price, color='r', linestyle='-', label='Last mean ETF investment value: %.3f ' % mean_final_price + '$') # plot the mean final price line
    plt.legend()
    # option to save figure or show plot depending on the value of variable save_fig
    if save_fig == 1:
        plt.savefig(str(name) + '.jpg', dpi = 300, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()
        plt.close()

# This function plots the histogram of the distribution of final values for the different simulations
def plot_histogram(name, results_simulation, color, save_fig = 1):       
    values = results_simulation.values
    final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe
    
    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(final_price, num_bins, density=True, alpha=0.5, label=name, color=color)

    #### Lines to be uncommented if wanna plot kde in this graph ####        
    #mu = final_price.mean()
    #sigma = final_price.std()
    #y = stats.norm.pdf(bins,mu,sigma)
    #plt.plot(bins, y, '--', label = 'pdf ' + str(name))

    plt.xlabel('Value ($)')
    plt.ylabel('Probability')
    plt.title(r'Histogram of Speculated Investment Value', fontsize=17, fontweight='bold')

    # plot vertical line of 30%-percentile 
    plt.axvline(np.percentile(final_price,30), linestyle='dashed', linewidth=2, label = '30%-ile: ' 
                + str(round(np.percentile(final_price,30), 2)) + '$', color='k')
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.legend()
    if save_fig == 1:
        plt.savefig(str(name) + '_hist.pdf', dpi = 300, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()
        plt.close()

# This function plots the histogram of the distribution of final values of the different models on the same graph
def plot_histogram_together(models, results_simulation, colors, save_fig):
    dict_names = list(models.keys()) # creating a list for the model names to be called as argument
    for i, (name, model), color in zip(range(len(results_simulation)), models.items(), colors): # looping over all resuls and arguments together     
        values = results_simulation[i].values
        final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe
    
        num_bins = 100
        # the histogram of the data
        n, bins, patches = plt.hist(final_price, num_bins, density=True, alpha=0.5, label=name, color=color)

    plt.xlabel('Value ($)')
    plt.ylabel('Probability')
    plt.title(r'Histogram of Speculated Investment Value', fontsize=17, fontweight='bold')
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if save_fig == 1:
        plt.savefig('hists_plot.pdf', dpi = 300, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()


# This function plots the kde of the distribution of final values for the simulation results
def plot_kde(name, results_simulation, color,save_fig = 1):       
    values = results_simulation.values
    final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe

    sns.distplot(final_price, rug=False, hist =False, 
                kde_kws={"color": str(color), "lw": 3, "shade": True, "label": str(name)})
                #hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})     

    plt.xlabel('Value ($)')
    plt.ylabel('Probability')
    plt.title(r'KDE of Speculated Investment Value', fontsize=17, fontweight='bold')
    plt.legend()

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if save_fig == 1:
        plt.savefig(str(name) + '_kde.pdf', dpi = 300, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()

# This function plots the kde of the different models on the same graph
def plot_kde_together(models, results_simulation, colors, save_fig = 1):
    dict_names = list(models.keys()) # creating a list for the model names to be called as argument
    for i, (name, model), color in zip(range(len(results_simulation)), models.items(), colors):  # looping over all resuls and arguments together     
        values = results_simulation[i].values
        final_price = values[-1]

        sns.distplot(final_price, rug=False, hist = False,
            kde_kws={"color": str(color), "alpha":0.5, 'linestyle':'--', "lw": 2, "shade": False, "label": str(name)})
                     

    plt.xlabel('Value ($)')
    plt.ylabel('Probability')
    plt.title(r'KDE of Speculated Investment Value', fontsize=17, fontweight='bold')
    plt.legend()

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    if save_fig == 1:
        plt.savefig('kdes_plot.pdf', dpi = 300, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()



# Plot the different histograms as subplot, such as in Fig 2. (alternate way) in the report
def plot_histograms_figure2(models, results_simulation, colors, save_fig = 1):       
    dict_names = list(models.keys()) # creating a list for the model names to be called as argument
    n_rows = len(results_simulation) - 1 # number of rows for subplots 
    n_cols = len(results_simulation) - 1 # number of cols for subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(9, 9), sharey=True, sharex=True)
    for  i in range(n_rows):
        for j in range(n_cols): #looping over rows and cols of the figure
            if j == i: #different tests to ensure the results from the right model are chosen, to plot only one curve for each subplot
                values = results_simulation[i].values
                final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe
                color = colors[i]
                name = dict_names[i] 
            elif j == i+1:
                values = results_simulation[i+1].values
                final_price = values[-1]
                color = colors[i+1]
                name = dict_names[i+1] 
            elif j == i-1:
                values = results_simulation[i+1].values
                final_price = values[-1] 
                color = colors[i+1]
                name = dict_names[i+1] 
            else:
                None
            num_bins = 100
            # the histogram of the data is plotted on each subplot
            n, bins, patches = axs[i,j].hist(final_price, num_bins, density=True, alpha=0.5, label=name, color=color)

            # plot vertical line of 30%-percentile 
            axs[i,j].axvline(np.percentile(final_price,30), linestyle='dashed', linewidth=2, label = '30%-ile: ' 
            + str(round(np.percentile(final_price,30), 2)) + '$', color='k')

            axs[i,j].set_xlabel('Value ($)')
            axs[i,j].set_ylabel('Probability')
            axs[i,j].xaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
            axs[i,j].legend()
    
    # The subplot on the lower right corner is cleaned as we want 3 curves over there    
    axs[1,1].clear()

    # Loop for the subplot on position (1,1) to get the 3 histograms on the same subplot
    for i, (name, model), color in zip(range(len(results_simulation)), models.items(), colors):
        values = results_simulation[i].values
        final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe
    
        num_bins = 100
        # the histogram of the data for subplot(1,1)
        n, bins, patches = axs[1,1].hist(final_price, num_bins, density=True, alpha=0.5, label=name, color=color)

        axs[1,1].set_xlabel('Value ($)')
        axs[1,1].set_ylabel('Probability')
        axs[1,1].xaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))

    plt.legend()
    fig.suptitle(r'Histograms of Speculated Investment Value', fontsize=18, fontweight='bold', y=0.92)
    if save_fig == 1:
        plt.savefig('Hists_fig2.pdf', dpi = 300, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()

# Plot the different kdes as subplot, such as in Fig 3. (alternate way) in the report
def plot_kdes_figure3(models, results_simulation, colors, save_fig = 1):       
    dict_names = list(models.keys()) # creating a list for the model names to be called as argument
    n_rows = len(results_simulation) - 1 # number of rows for subplots 
    n_cols = len(results_simulation) - 1 # number of cols for subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(9, 9), sharey=True, sharex=True)
    for  i in range(n_rows):
        for j in range(n_cols): #looping over rows and cols of the figure
            if j == i: #different tests to ensure the results from the right model are chosen, to plot only one curve for each subplot
                values = results_simulation[i].values
                final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe
                color = colors[i] 
                name = dict_names[i] 
            elif j == i+1:
                values = results_simulation[i+1].values
                final_price = values[-1]
                color = colors[i+1]
                name = dict_names[i+1] 
            elif j == i-1:
                values = results_simulation[i+1].values
                final_price = values[-1] 
                color = colors[i+1]
                name = dict_names[i+1] 
            else:
                None

            # the kde of the data is plotted on each subplot
            sns.distplot(final_price, rug=False, hist =False, ax=axs[i,j],
            kde_kws={"color": str(color), "lw": 3, "shade": True, "label": str(name)})
            #hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})   


            axs[i,j].set_xlabel('Value ($)')
            axs[i,j].set_ylabel('Probability')
            axs[i,j].xaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
            axs[i,j].legend()

    # The subplot on the lower right corner is cleaned as we want 3 curves over there   
    axs[1,1].clear()

    # Loop for the subplot on position (1,1) to get the 3 histograms on the same subplot
    for i, (name, model), color in zip(range(len(results_simulation)), models.items(), colors):
        values = results_simulation[i].values
        final_price = values[-1] # retrieve the final price data --> values of last row of the results dataframe
    
        # kde of the data for subplot(1,1)
        sns.distplot(final_price, rug=False, hist =False, ax = axs[1,1], 
        kde_kws={"color": str(color), "alpha":0.5, 'linestyle':'--', "lw": 2, "shade": False, "label": str(name)})
        #hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})   

        axs[1,1].set_xlabel('Value ($)')
        axs[1,1].set_ylabel('Probability')
        axs[1,1].xaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        #axs[1,1].ticklabel_format(style='sci', axis='x', scilimits=(4,0))

    plt.legend()
    fig.suptitle(r'KDEs of Speculated Investment Value', fontsize=18, fontweight='bold', y=0.92)
    plt.savefig('KDE_Fig3.pdf', dpi = 300, bbox_inches='tight')



# This function prints different statistics for the simulation results
def show_stats(name, results_simulation):
    print('#----------------------Last Price Stats ' + str(name) + '--------------------#')
    print("Mean Price: ", np.mean(results_simulation.iloc[-1,:]))
    print("Maximum Price: ",np.max(results_simulation.iloc[-1,:]))
    print("Minimum Price: ", np.min(results_simulation.iloc[-1,:]))
    print("Standard Deviation: ",np.std(results_simulation.iloc[-1,:]))
 
    print()
       
    print('#----------------------Descriptive Stats ' + str(name) + '-------------------#')
    price_array = results_simulation.iloc[-1, :]
    print(price_array.describe())

    print('\n')

# This function prints the percentiles for the final price distribution of the simulation
def print_percentiles(name, results_simulation):
    values = results_simulation.values
    final_price = values[-1]
    p_tiles = np.percentile(final_price,[5,15,30,50,70,85,95])
    print('Percentiles for the distribution of ' + str(name))
    for p in range(len(p_tiles)):
        l = [5,15,30,50,70,85,95]
        print( "{}%-ile: ".format(l[p]).rjust(15),"{}".format(round(p_tiles[p],1), grouping=True) + '$')
    print()
    
    print('There is 70% chance that the initial investment after 10 years is superior to', str(round(p_tiles[2],1)) + '$')
    print()
    print('There is 70% chance that the initial investment after 10 years is in the range ]'
    + str(round(p_tiles[1],1)) + ',' + str(round(p_tiles[-2],1)) + ']')

    print('\n')
