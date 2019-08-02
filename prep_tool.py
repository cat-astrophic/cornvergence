# This script takes the data from NASS and converts it to a more appropriate format for this study

# Importing required modules

import pandas as pd
import numpy as np

# Declaring the files requiring preparation

files = ['C:/Users/User/Documents/Data/Cornvergence/Corn_Counties_Yield_Belt_NASS.csv', 'C:/Users/User/Documents/Data/Cornvergence/Corn_Counties_Yield_All_NASS.csv']

# Defining the prep function

def data_prep(file):
    
    # Loading the data
    
    data = pd.read_csv(file)
    
    # Creating a new reference column and deleting unwanted columns
    
    data = data[['Year', 'State', 'County', 'Commodity', 'Data Item', 'Value']]
    new_col = [str(data.County[i]) + ', ' + str(data.State[i]) for i in range(len(data.County))]
    new_col = pd.DataFrame(new_col, columns = ['Index'])
    data = pd.concat([new_col, data], axis = 1)
    
    # Summarizing the data to see which counties have data for every year  of the study (15)
    
    table = data.groupby('Index').count()
    listy = [table.index[i] for i in range(len(table.Year)) if table.Year[i] == 25]
    reflist = [i for i in range(len(data.Index)) if data.Index[i] in listy]
    
    # Subsetting the dataframe based on above results and separating yield and acres planted data
    
    data = data.iloc[reflist]
    yield_df = data[data['Data Item'] == 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE']
    yield_df = yield_df.sort_values(by = ['Index', 'Year'])
    
    # Creating the time series for convergence testing
    
    counties = yield_df.Index.unique()
    df = pd.DataFrame(columns = [i for i in range(1991,2016)])
    
    for county in counties:
        temp = yield_df[yield_df['Index'] == county]
        temp_list = np.transpose([val for val in temp.Value.astype(float)]).reshape(1,25)
        temp_list = pd.DataFrame(temp_list, columns = df.columns)
        df = pd.concat([df, temp_list], axis = 0)
    
    df = df.set_index(counties)
    df = df.reset_index()
    df = df.rename(columns={'index':'County'})
    df = df.sort_values(by = df.columns[len(df.columns)-1], ascending = False)
    output = file[0:len(file)-9] + '_TimeSeries.csv'
    df.to_csv(output, index = False)

# Using the data_prep function to prep each data set

for file in files:
    data_prep(file)

