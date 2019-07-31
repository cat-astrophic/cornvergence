# This is the (main) script which runs the beta and club convergence tests
# Before running this, you should download the data from NASS, per the instructions in the paper, and run prep_tool.py

# Importing required modules

import pandas as pd
import numpy as np
import statsmodels.api as stats
from matplotlib import pyplot as plt

# Importing the time series data

data_all = pd.read_csv('C:/Users/User/Documents/Data/Corn_Counties_Yield_All_TimeSeries.csv')
data_belt = pd.read_csv('C:/Users/User/Documents/Data/Corn_Counties_Yield_Belt_TimeSeries.csv')

# Creating the dataframes for the beta convergence test

df_belt = pd.DataFrame(columns = ['County', 'Rate', 'Initial'])
for i in range(len(data_belt.County)):
    rate = np.log(data_belt['2015'][i] / data_belt['1991'][i])
    init = np.log(data_belt['1991'][i])
    row = np.transpose([data_belt.County[i], rate, init]).reshape(1,3)
    row = pd.DataFrame(row, columns = df_belt.columns)
    df_belt = pd.concat([df_belt, row], axis = 0)

df_all = pd.DataFrame(columns = ['County', 'Rate', 'Initial'])
for i in range(len(data_all.County)):
    rate = np.log(data_all['2015'][i] / data_all['1991'][i])
    init = np.log(data_all['1991'][i])
    row = np.transpose([data_all.County[i], rate, init]).reshape(1,3)
    row = pd.DataFrame(row, columns = df_all.columns)
    df_all = pd.concat([df_all, row], axis = 0)

Y_belt = df_belt.Rate.astype(float) / 24
Y_belt = pd.DataFrame(Y_belt, columns = ['Rate'])
X_belt = df_belt['Initial']
X_belt = stats.add_constant(X_belt)

Y_all = df_all.Rate.astype(float) / 24
Y_all = pd.DataFrame(Y_all, columns = ['Rate'])
X_all = df_all['Initial']
X_all = stats.add_constant(X_all)

# Performing the beta convergence test and recording results

model_belt = stats.OLS(Y_belt.astype(float), X_belt.astype(float))
results_belt = model_belt.fit()
print(results_belt.summary())
file = open('C:/Users/User/Documents/Data/beta_convergence_test_results_belt.txt', 'w')
file.write(results_belt.summary().as_text())
file.close()

threshold = 0.05
if (results_belt.params.Initial < 0) and (results_belt.pvalues.Initial < threshold):
    print('\n---> There is sufficient evidence to support the hypothesis of beta convergence in the corn belt! <---\n')

model_all = stats.OLS(Y_all.astype(float), X_all.astype(float))
results_all = model_all.fit()
print(results_all.summary())
file = open('C:/Users/User/Documents/Data/beta_convergence_test_results_all.txt', 'w')
file.write(results_all.summary().as_text())
file.close()

threshold = 0.05
if (results_all.params.Initial < 0) and (results_all.pvalues.Initial < threshold):
    print('\n---> There is sufficient evidence to support the hypothesis of beta convergence everywhere! <---\n')

# Drawing the scatter plots

x_belt = X_belt['Initial'].values.astype(float)
y_belt = Y_belt['Rate'].values.astype(float)
plt.figure(figsize = (6,5))
plt.scatter(x_belt, y_belt, marker = '.', color = 'black')
plt.xlabel('Initial Level (Ln)')
plt.ylabel('Growth Rate')
plt.title('Yield Growth Rate as a function of Initial Yield', fontsize = 13)
plt.savefig('C:/Users/User/Documents/Data/beta_convergence_plot_belt.eps')
xx_belt = stats.add_constant(x_belt)
mod_belt = stats.OLS(y_belt, xx_belt)
res_belt = mod_belt.fit()
b = res_belt.params[0]
m = res_belt.params[1]
line = m * x_belt + b
plt.plot(x_belt, line, color = 'black')
plt.savefig('C:/Users/User/Documents/Data/beta_convergence_plot_belt_line.eps')

x_all = X_all['Initial'].values.astype(float)
y_all = Y_all['Rate'].values.astype(float)
plt.figure(figsize = (6,5))
plt.scatter(x_all, y_all, marker = '.', color = 'red')
plt.scatter(x_belt, y_belt, marker = '.', color = 'black')
plt.xlabel('Initial Level (Ln)')
plt.ylabel('Growth Rate')
plt.title('Yield Growth Rate as a function of Initial Yield', fontsize = 13)
plt.savefig('C:/Users/User/Documents/Data/beta_convergence_plot_all.eps')
xx_all = stats.add_constant(x_all)
mod_all = stats.OLS(y_all, xx_all)
res_all = mod_all.fit()
b = res_all.params[0]
m = res_all.params[1]
line2 = m * x_all + b
plt.plot(x_belt, line, color = 'black')
plt.plot(x_all, line2, color = 'red')
plt.savefig('C:/Users/User/Documents/Data/beta_convergence_plot_all_line.eps')

# Now we move on to sigma convergence by calculating annual sample standard deviations and coefficients of variation

sigma_belt = [np.std(data_belt[str(i)]) for i in range(1991,2016)]
cv_belt = [np.std(data_belt[str(i)]) / np.mean(data_belt[str(i)]) for i in range(1991,2016)]

sigma_all = [np.std(data_all[str(i)]) for i in range(1991,2016)]
cv_all = [np.std(data_all[str(i)]) / np.mean(data_all[str(i)]) for i in range(1991,2016)]

# Write sigma convergence results to file







































