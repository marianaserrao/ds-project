import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns

# ______________ General infos and changes ______________ 
# drought = class 
# type of registries: binary
# 0 - no drought
# 1 - drought
# ______________________________________________________
# The KDD Process should be follow, here are the steps
# ______________________________________________________
# 1. define the goal: give an entry, classify if there is drought or not (binary output)
# 2. collect the data: already done. there is nothing to add.

data = pd.read_csv('../drought.csv')

# to be clear about the output variable name, the change from "class" to "drought" is applied

data = data.rename(columns = {"class" : "drought"})

# 3. explore the data/data profiling:
#   3.1. Data caracteristics
#    about fips code: https://www.weather.gov/pimar/FIPSCodes

data.info()
# the date attribute is a object, so transforming into separate columns:
data['day'] = pd.DatetimeIndex(data['date'], dayfirst=True).day
data['month'] = pd.DatetimeIndex(data['date'], dayfirst=True).month
data['year'] = pd.DatetimeIndex(data['date'], dayfirst=True).year
# rearrange of the columns order just to best visualize on heatmap

data = data[[col for col in data if col not in ['drought']] + ['drought']]

# removing the date column  
data.pop('date')

# unique values

uniqueValues = data['fips'].nunique()

#   3.2. Distribution:
#     3.2.1. Missing values (MV)

nullValues = data.isnull().sum()

#     there is no missing values

#     3.2.3. Histograms

def histograms(data, dimension):
    i, j = dimension
    fig, ax = plt.subplots(i, j, figsize=(50, 80))
    
    for position in range(len(data.columns)):
        col = data.columns[position]

        pos_i = position//j
        pos_j = position%j
        

        dist_0 = data[data['drought']==0][col]
        dist_1 = data[data['drought']==1][col]

        ax[pos_i][pos_j].hist([dist_0, dist_1],
                          stacked=False,
                          label=['drought = 0', 'drought = 1'],
                          color=['#7547B8', '#8AB847'])
        ax[pos_i][pos_j].set_title(col)
        ax[pos_i][pos_j].legend()

    plt.savefig('drought_hist01.png')

#histograms(data, (9, 6))
#    


#   3.1. Granularity:
#   3.3. Sparsity:

def boxplot(data, filename):

    #data=pd.melt(data, var_name='Attributes', value_name='Values'
    fig, ax = plt.subplots()
    print(data)
    sns.boxplot(data=data)
    plt.xticks(rotation='vertical')
    plt.savefig(filename + '.png')

def setBoxplotInterval(data=pd.DataFrame, tupleIntervals=[tuple]):
    """ The function receives the data as dataframe and a list of 
        tuples that represents the intervals to filter the data.
        It's considered that the first element of the tuple is the 
        initial value of the interval and the second element is the
        end value of the interval. It's also suposed that there is 
        no conflict between the tuples of the list. """

    dataDescript = data.describe().transpose()
    print(dataDescript)
    dfsIntervals = []
    for entry in tupleIntervals: 
        tmpDF = dataDescript[(dataDescript['min']>=entry[0]) & dataDescript[(dataDescript['min']<entry[1]) & (dataDescript['max']>=entry[2])] & (dataDescript['max']<entry[3])] 
        if len(tmpDF.transpose().columns) != 0:
            dfsIntervals.append(tmpDF.transpose())
        else: continue

    return dfsIntervals

boxplotIntervals = setBoxplotInterval(data, [(-100, 0), (0, 1), (0, 10), (0, 50), (0, 120), (0, 1000), (0, 1999), (0, 2017)])
labelFilename = 0
for inter in boxplotIntervals:
    boxplot(data[inter.columns], 'boxplot'+str(labelFilename))
    labelFilename+=1
 

#boxplot(dfs, 'drought_boxplot')
#boxplot(tmp_data, 'drought_boxplot')

def heatmap(data):
    heatmap = data.corr()
    f, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(heatmap,
                cmap=sns.color_palette("RdBu_r", 1000),
                vmin=-1,
                vmax=1,
                square=True)
    
    plt.savefig('drought_heatmap01.png')

#heatmap(data)

#   3.4. Dimensionality:
# 4. prepare the data:
# 5. learn the model:
# 6. delirevy the model: 
# step 1: change from class to drought
# _____________________________
# ----------- Classification -----------
# Remove: flips and date because dont aggregate anything

