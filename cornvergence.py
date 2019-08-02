# This is the (main) script which runs the beta and club convergence tests
# Before running this, you should download the data from NASS, per the instructions in the paper, and run prep_tool.py

# Importing required modules

import pandas as pd
import numpy as np
import statsmodels.api as stats
from matplotlib import pyplot as plt
import plotly.figure_factory as ff

# Importing the time series data

data_all = pd.read_csv('C:/Users/User/Documents/Data/Cornvergence/Corn_Counties_Yield_All_TimeSeries.csv')
data_belt = pd.read_csv('C:/Users/User/Documents/Data/Cornvergence/Corn_Counties_Yield_Belt_TimeSeries.csv')

# Creating counties directories for future use

counties_all = data_all.County
counties_belt = data_belt.County

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
file = open('C:/Users/User/Documents/Data/Cornvergence/beta_convergence_test_results_belt.txt', 'w')
file.write(results_belt.summary().as_text())
file.close()

threshold = 0.05
if (results_belt.params.Initial < 0) and (results_belt.pvalues.Initial < threshold):
    print('\n---> There is sufficient evidence to support the hypothesis of beta convergence in the corn belt! <---\n')

model_all = stats.OLS(Y_all.astype(float), X_all.astype(float))
results_all = model_all.fit()
print(results_all.summary())
file = open('C:/Users/User/Documents/Data/Cornvergence/beta_convergence_test_results_all.txt', 'w')
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
plt.savefig('C:/Users/User/Documents/Data/Cornvergence/beta_convergence_plot_belt.eps')
xx_belt = stats.add_constant(x_belt)
mod_belt = stats.OLS(y_belt, xx_belt)
res_belt = mod_belt.fit()
b = res_belt.params[0]
m = res_belt.params[1]
line = m * x_belt + b
plt.plot(x_belt, line, color = 'black')
plt.savefig('C:/Users/User/Documents/Data/Cornvergence/beta_convergence_plot_belt_line.eps')

x_all = X_all['Initial'].values.astype(float)
y_all = Y_all['Rate'].values.astype(float)
plt.figure(figsize = (6,5))
plt.scatter(x_all, y_all, marker = '.', color = 'red')
plt.scatter(x_belt, y_belt, marker = '.', color = 'black')
plt.xlabel('Initial Level (Ln)')
plt.ylabel('Growth Rate')
plt.title('Yield Growth Rate as a function of Initial Yield', fontsize = 13)
plt.savefig('C:/Users/User/Documents/Data/Cornvergence/beta_convergence_plot_all.eps')
xx_all = stats.add_constant(x_all)
mod_all = stats.OLS(y_all, xx_all)
res_all = mod_all.fit()
b = res_all.params[0]
m = res_all.params[1]
line2 = m * x_all + b
plt.plot(x_belt, line, color = 'black')
plt.plot(x_all, line2, color = 'red')
plt.savefig('C:/Users/User/Documents/Data/Cornvergence/beta_convergence_plot_all_line.eps')

# Now we move on to sigma convergence by calculating annual sample standard deviations and coefficients of variation

sigma_belt = [np.std(data_belt[str(i)]) for i in range(1991,2016)]
cv_belt = [np.std(data_belt[str(i)]) / np.mean(data_belt[str(i)]) for i in range(1991,2016)]

sigma_all = [np.std(data_all[str(i)]) for i in range(1991,2016)]
cv_all = [np.std(data_all[str(i)]) / np.mean(data_all[str(i)]) for i in range(1991,2016)]

# Write sigma convergence results to file

sigmastats = [['Corn Belt - STD', sigma_belt[0] - sigma_belt[len(sigma_belt)-1]], ['Corn Belt - CV', cv_belt[0] - cv_belt[len(cv_belt)-1]], ['All - STD', sigma_all[0] - sigma_all[len(sigma_all)-1]], ['All - CV', cv_all[0] - cv_all[len(cv_all)-1]]]
cols = ['Measure', 'Difference']
for i in range(1991,2016):
    cols.append(str(i))
    sigmastats[0].append(sigma_belt[i-1991])
    sigmastats[1].append(cv_belt[i-1991])
    sigmastats[2].append(sigma_all[i-1991])
    sigmastats[3].append(cv_all[i-1991])
sigma_df = pd.DataFrame(sigmastats, columns = cols)
sigma_df.to_csv('C:/Users/User/Documents/Data/Cornvergence/sigma_stats.txt', index = False)

# Club Convergence (via Phillips and Sul, 2007)

# Resetting the index for convenience

data_all = data_all.set_index('County')
data_belt = data_belt.set_index('County')

# First, create the matrix X_{it}

years = [i for i in range(1991, 2016)]
vals_all = [sum(data_all[str(year)]) for year in years]
vals_belt = [sum(data_belt[str(year)]) for year in years]

little_h_all = np.zeros(np.shape(data_all))
for i in range(len(data_all)):
    for j in range(len(years)):
        little_h_all[i,j] = data_all.values[i,j] / ((1 / len(data_all)) * vals_all[j])

little_h_belt = np.zeros(np.shape(data_belt))
for i in range(len(data_belt)):
    for j in range(len(years)):
        little_h_belt[i,j] = data_belt.values[i,j] / ((1 / len(data_belt)) * vals_belt[j])

# Second, find the cross sectional variance of X_{it}

BIG_H_ALL = np.zeros((1, len(years)))
BIG_H_BELT = np.zeros((1, len(years)))
for i in range(len(years)):
    s_all = 0
    s_belt = 0
    for j in range(len(data_all)):
        s_all += (little_h_all[j,i] - 1) ** 2
    BIG_H_ALL[0,i] = s_all / len(data_all)
    for j in range(len(data_belt)):
        s_belt += (little_h_belt[j,i] - 1) ** 2
    BIG_H_BELT[0,i] = s_belt / len(data_belt)

# Third, run regression to obtain estiamtes \hat{a} and \hat{b} -- we choose var = 5 based on observing sigma_all and sigma_belt

var = 5
ratios_all = [np.log(BIG_H_ALL[0,0] / BIG_H_ALL[0,t]) for t in range(var,len(years))]
ratios_belt = [np.log(BIG_H_BELT[0,0] / BIG_H_BELT[0,t]) for t in range(var,len(years))]

LHS_all = [ratios_all[t] - 2*(np.log(np.log(t+var))) for t in range(len(ratios_all))]
LHS_all = pd.DataFrame(LHS_all, columns = ['LHS'])
RHS_all = [np.log(t) for t in range(var,len(years))]
RHS_all = pd.DataFrame(RHS_all, columns = ['RHS'])
RHS_all = stats.add_constant(RHS_all)

LHS_belt = [ratios_belt[t] - 2*(np.log(np.log(t+var))) for t in range(len(ratios_belt))]
LHS_belt = pd.DataFrame(LHS_belt, columns = ['LHS'])
RHS_belt = [np.log(t) for t in range(var,len(years))]
RHS_belt = pd.DataFrame(RHS_belt, columns = ['RHS'])
RHS_belt = stats.add_constant(RHS_belt)

club_model_all = stats.OLS(LHS_all, RHS_all)
club_results_all = club_model_all.fit()
print(club_results_all.summary())
file = open('C:/Users/User/Documents/Data/Cornvergence/club_results_all.txt', 'w')
file.write(club_results_all.summary().as_text())
file.close()

club_model_belt = stats.OLS(LHS_belt, RHS_belt)
club_results_belt = club_model_belt.fit()
print(club_results_belt.summary())
file = open('C:/Users/User/Documents/Data/Cornvergence/club_results_belt.txt', 'w')
file.write(club_results_belt.summary().as_text())
file.close()

# Given the results for \hat{a} and \hat{b}, we may use a club convergence algorithm to determine club membership

# We define a function for determining convergence club membership

def clubbing(idx, dataset):
    
    club_vals = [sum(dataset[str(year)][0:idx+1]) for year in years]
    
    club_h = np.zeros(np.shape(dataset))
    for i in range(idx):
        for j in range(len(years)):
            club_h[i,j] = dataset.values[i,j] / ((1 / (idx+1)) * club_vals[j])
    
    club_H = np.zeros((1, len(years)))
    for i in range(len(years)):
        s = 0
        for j in range(idx):
            s += (club_h[j,i] - 1) ** 2
        club_H[0,i] = s / idx
    
    var = 5
    ratios = [np.log(club_H[0,0] / club_H[0,t]) for t in range(var,len(years))]
    LHS = [ratios[t] - 2*(np.log(np.log(t+var))) for t in range(len(ratios))]
    LHS = pd.DataFrame(LHS, columns = ['LHS'])
    RHS = [np.log(t) for t in range(var,len(years))]
    RHS = pd.DataFrame(RHS, columns = ['RHS'])
    RHS = stats.add_constant(RHS)
    
    club_model = stats.OLS(LHS, RHS)
    club_results = club_model.fit()
    beta = club_results.params[1]
    return beta

# Creating the convergence clubs

# Plots of the ranked final year ordered yields

basis_all = [i for i in range(1,len(data_all)+1)]
plt.figure(figsize = (6,5))
plt.plot(basis_all, data_all['2015'], color = 'black')
plt.xlabel('County Rank')
plt.ylabel('Yield (BU / ACRE)')
plt.title('Ordered Plot of County Yields - All', fontsize = 13)
plt.savefig('C:/Users/User/Documents/Data/Cornvergence/2015_plot_all.eps')

basis_belt = [i for i in range(1,len(data_belt)+1)]
plt.figure(figsize = (6,5))
plt.plot(basis_belt, data_belt['2015'], color = 'black')
plt.xlabel('County Rank')
plt.ylabel('Yield (BU / ACRE)')
plt.title('Ordered Plot of County Yields - Corn Belt', fontsize = 13)
plt.savefig('C:/Users/User/Documents/Data/Cornvergence/2015_plot_belt.eps')

# Club membership algorithm using the function above - all counties

idx = 0
a = 0
clubs = []
remaining = [i for i in range(len(data_all))]
while len(remaining) > 0:
    beta = 1
    while beta > 0:
        idx += 1
        print(a+idx)
        if (a + idx) == max(remaining):
            clubs.append(remaining)
            remaining = []
            beta = -1
        else:
            beta = clubbing(idx, data_all)
            if beta < 0:
                club = [i for i in range(a,a+idx)]
                clubs.append(club)
                for item in club:
                    remaining.remove(item)

    data_all = data_all.iloc[idx:len(data_all), :]
    print(data_all)
    a += idx
    idx = 0

# List names of members of clubs

all_clubs = clubs
all_members = []
for club in clubs:
    group = [counties_all[idx] for idx in club]
    all_members.append(group)

# Club membership algorithm using the function above - corn belt
    
idx = 1
a = 0
clubs = []
remaining = [i for i in range(len(data_belt))]
while len(remaining) > 0:
    beta = 1
    while beta > 0:
        idx += 1
        print(a+idx)
        if (a + idx) == max(remaining):
            clubs.append(remaining)
            remaining = []
            beta = -1
        else:
            beta = clubbing(idx, data_belt)
            if beta < 0:
                if idx == 2:
                    idx = 1
                club = [i for i in range(a,a+idx)]
                clubs.append(club)
                for item in club:
                    remaining.remove(item)

    data_belt = data_belt.iloc[idx:len(data_belt), :]
    print(data_belt)
    a += idx
    idx = 1

# List names of members of clubs

belt_clubs = clubs
belt_members = []
for club in clubs:
    group = [counties_belt[idx] for idx in club]
    belt_members.append(group)

# Write clubs with members to txt file for reference

all_members = pd.DataFrame(all_members)
belt_members = pd.DataFrame(belt_members)
all_members.to_csv('C:/Users/User/Documents/Data/Cornvergence/club_membership_all.csv', index = False, header = False)
belt_members.to_csv('C:/Users/User/Documents/Data/Cornvergence/club_membership_belt.csv', index = False, header = False)

# Chloropleths

# Chloropleth of relative yields for corn belt counties - 2015

counties_belt = list(counties_belt.values)
belt_data = pd.read_csv('C:/Users/User/Documents/Data/Cornvergence/Corn_Counties_Yield_Belt_TimeSeries.csv')
belt_states = ['Illinois', 'Indiana', 'Iowa', 'Kansas', 'Michigan', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'Ohio', 'South Dakota', 'Wisconsin']
df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
cdf = df_sample[df_sample['STNAME'].isin(belt_states)][['FIPS', 'STNAME', 'CTYNAME']]
cdf.index = pd.RangeIndex(len(cdf.index))

for i in range(len(cdf.CTYNAME)):
    cdf.CTYNAME[i] = cdf.CTYNAME[i][0:len(cdf.CTYNAME[i])-7] + ', ' + cdf.STNAME[i]
    cdf.CTYNAME[i] = cdf.CTYNAME[i].upper()

last_vals = []
for city in cdf.CTYNAME:
    try:
        idx = counties_belt.index(city)
        last_vals.append(belt_data['2015'][idx])
    except:
        last_vals.append(0)
        pass

lv = pd.DataFrame(last_vals, columns = ['2015'])
cdf = pd.concat([cdf, lv], axis = 1)

values = cdf['2015'].tolist()
fips = cdf['FIPS'].tolist()

binning_endpoints = [i*10 for i in range(24)]
cs = [
      'rgb(255,255,255)',
      'rgb(250,250,250)',
      'rgb(240,240,240)',
      'rgb(230,230,230)',
      'rgb(220,220,220)',
      'rgb(210,210,210)',
      'rgb(200,200,200)',
      'rgb(190,190,190)',
      'rgb(180,180,180)',
      'rgb(170,170,170)',
      'rgb(160,160,160)',
      'rgb(150,150,150)',
      'rgb(140,140,140)',
      'rgb(130,130,130)',
      'rgb(120,120,120)',
      'rgb(110,110,110)',
      'rgb(100,100,100)',
      'rgb(90,90,90)',
      'rgb(80,80,80)',
      'rgb(70,70,70)',
      'rgb(60,60,60)',
      'rgb(50,50,50)',
      'rgb(40,40,40)',
      'rgb(30,30,30)',
      'rgb(20,20,20)',
      'rgb(0,0,0)'
]

fig = ff.create_choropleth(
    fips = fips, values = values,
    scope = belt_states, county_outline = {'color': 'rgb(255,255,255)', 'width': 0.5},
    binning_endpoints = binning_endpoints, 
    colorscale = cs,
    legend_title = 'Yield (BU / ACRE) by County'
)

fig.update_layout(
    legend_x = 0,
    annotations = {'x': -0.12, 'xanchor': 'left'}
)

fig.layout.template = None
fig.write_image('C:/Users/User/Documents/Data/Cornvergence/belt_scale_2015.png')

# Chloropleth of relative yields for corn belt counties - 1991

cdf = df_sample[df_sample['STNAME'].isin(belt_states)][['FIPS', 'STNAME', 'CTYNAME']]
cdf.index = pd.RangeIndex(len(cdf.index))

for i in range(len(cdf.CTYNAME)):
    cdf.CTYNAME[i] = cdf.CTYNAME[i][0:len(cdf.CTYNAME[i])-7] + ', ' + cdf.STNAME[i]
    cdf.CTYNAME[i] = cdf.CTYNAME[i].upper()

last_vals = []
for city in cdf.CTYNAME:
    try:
        idx = counties_belt.index(city)
        last_vals.append(belt_data['1991'][idx])
    except:
        last_vals.append(0)
        pass

lv = pd.DataFrame(last_vals, columns = ['1991'])
cdf = pd.concat([cdf, lv], axis = 1)

values2 = cdf['1991'].tolist()
fips = cdf['FIPS'].tolist()

fig = ff.create_choropleth(
    fips = fips, values = values2,
    scope = belt_states, county_outline = {'color': 'rgb(255,255,255)', 'width': 0.5},
    binning_endpoints = binning_endpoints, 
    colorscale = cs,
    legend_title = 'Yield (BU / ACRE) by County'
)

fig.update_layout(
    legend_x = 0,
    annotations = {'x': -0.12, 'xanchor': 'left'}
)

fig.layout.template = None
fig.write_image('C:/Users/User/Documents/Data/Cornvergence/belt_scale_1991.png')

# Difference in yeilds by county - corn belt

be2 = [i*10 for i in range(15)]
cs2 = [
      'rgb(255,255,255)',
      'rgb(240,240,240)',
      'rgb(225,225,225)',
      'rgb(210,210,210)',
      'rgb(195,195,195)',
      'rgb(180,180,180)',
      'rgb(165,165,165)',
      'rgb(150,150,150)',
      'rgb(135,135,135)',
      'rgb(120,120,120)',
      'rgb(105,105,105)',
      'rgb(90,90,90)',
      'rgb(75,75,75)',
      'rgb(60,60,60)',
      'rgb(45,45,45)',
      'rgb(30,30,30)',
      'rgb(0,0,0)'
]

diffs = [values[i] - values2[i] for i in range(len(values))]
diffs = pd.DataFrame(diffs, columns = ['Difference'])
vals = diffs['Difference'].tolist()

fig = ff.create_choropleth(
    fips = fips, values = vals,
    scope = belt_states, county_outline = {'color': 'rgb(255,255,255)', 'width': 0.5},
    binning_endpoints = be2, 
    colorscale = cs2,
    legend_title = 'Change in Yield (BU / ACRE) by County<br>1991 - 2015'
)

fig.update_layout(
    legend_x = 0,
    annotations = {'x': -0.12, 'xanchor': 'left'}
)

fig.layout.template = None
fig.write_image('C:/Users/User/Documents/Data/Cornvergence/belt_diffs.png')

# Same for all counties - 2015

counties_all = list(counties_all.values)
all_data = pd.read_csv('C:/Users/User/Documents/Data/Cornvergence/Corn_Counties_Yield_All_TimeSeries.csv')

all_states = ['Alabama','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Islnad','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
cdf = df_sample[df_sample['STNAME'].isin(all_states)][['FIPS', 'STNAME', 'CTYNAME']]
cdf.index = pd.RangeIndex(len(cdf.index))

for i in range(len(cdf.CTYNAME)):
    cdf.CTYNAME[i] = cdf.CTYNAME[i][0:len(cdf.CTYNAME[i])-7] + ', ' + cdf.STNAME[i]
    cdf.CTYNAME[i] = cdf.CTYNAME[i].upper()

last_vals = []
for city in cdf.CTYNAME:
    try:
        idx = counties_all.index(city)
        last_vals.append(all_data['2015'][idx])
    except:
        last_vals.append(0)
        pass

lv = pd.DataFrame(last_vals, columns = ['2015'])
cdf = pd.concat([cdf, lv], axis = 1)

values = cdf['2015'].tolist()
fips = cdf['FIPS'].tolist()

fig = ff.create_choropleth(
    fips = fips, values = values,
    scope = all_states, county_outline = {'color': 'rgb(255,255,255)', 'width': 0.5},
    binning_endpoints = binning_endpoints, 
    colorscale = cs,
    legend_title = 'Yield (BU / ACRE) by County'
)

fig.update_layout(
    legend_x = 0,
    annotations = {'x': -0.12, 'xanchor': 'left'}
)

fig.layout.template = None
fig.write_image('C:/Users/User/Documents/Data/Cornvergence/all_scale_2015.png')

# Same for all counties - 1991

cdf = df_sample[df_sample['STNAME'].isin(all_states)][['FIPS', 'STNAME', 'CTYNAME']]
cdf.index = pd.RangeIndex(len(cdf.index))

for i in range(len(cdf.CTYNAME)):
    cdf.CTYNAME[i] = cdf.CTYNAME[i][0:len(cdf.CTYNAME[i])-7] + ', ' + cdf.STNAME[i]
    cdf.CTYNAME[i] = cdf.CTYNAME[i].upper()

last_vals = []
for city in cdf.CTYNAME:
    try:
        idx = counties_all.index(city)
        last_vals.append(all_data['1991'][idx])
    except:
        last_vals.append(0)
        pass

lv = pd.DataFrame(last_vals, columns = ['1991'])
cdf = pd.concat([cdf, lv], axis = 1)

values2 = cdf['1991'].tolist()
fips = cdf['FIPS'].tolist()

fig = ff.create_choropleth(
    fips = fips, values = values2,
    scope = all_states, county_outline = {'color': 'rgb(255,255,255)', 'width': 0.5},
    binning_endpoints = binning_endpoints, 
    colorscale = cs,
    legend_title = 'Yield (BU / ACRE) by County'
)

fig.update_layout(
    legend_x = 0,
    annotations = {'x': -0.12, 'xanchor': 'left'}
)

fig.layout.template = None
fig.write_image('C:/Users/User/Documents/Data/Cornvergence/all_scale_1991.png')

# Difference in yeilds by county - all

diffs = [values[i] - values2[i] for i in range(len(values))]
diffs = pd.DataFrame(diffs, columns = ['Difference'])
vals = diffs['Difference'].tolist()

fig = ff.create_choropleth(
    fips = fips, values = vals,
    scope = all_states, county_outline = {'color': 'rgb(255,255,255)', 'width': 0.5},
    binning_endpoints = be2, 
    colorscale = cs2,
    legend_title = 'Change in Yield (BU / ACRE) by County<br>1991 - 2015'
)

fig.update_layout(
    legend_x = 0,
    annotations = {'x': -0.12, 'xanchor': 'left'}
)

fig.layout.template = None
fig.write_image('C:/Users/User/Documents/Data/Cornvergence/all_diffs.png')

