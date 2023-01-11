from cmath import pi
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection  import train_test_split 
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import accuracy_score
from sklearn.naive_bayes      import GaussianNB
from sklearn.preprocessing    import StandardScaler, Normalizer

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

#   2.1. Data caracteristics
#    about fips code: https://www.weather.gov/pimar/FIPSCodes

# 3. explore the data/data profiling:
#   3.1. Granularity

# the date attribute is a object containing the timestamp. to increase the granularity, it's possible to 
# create an attribute for each field of the date attribute. the date attribute can be separated as day, month and year.
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
#print(uniqueValues)

drought1 = data['drought'].value_counts()
#print(drought1)
# the data is balacend

dtypesCount = data.dtypes
dfDtypesCount = pd.DataFrame(dtypesCount)
sumDtypes = dfDtypesCount.value_counts()
#print(sumDtypes)

fipsCount = data['fips'].value_counts()
#print(fipsCount)
# the proportion of entries per fips is standard (n=887)

#   3.2. Distribution:
#     3.2.1. Missing values (MV)

nullValues = data.isnull().sum()
#print(nullValues)

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

    plt.savefig('plots/drought_hist01.png')

#histograms(data, (9, 6))
#    
#     3.2.4. Boxplots

"""def boxplots(data):
    """"""

    labelFilename = 0
    for d in data:
        fig, ax = plt.subplots()
        sns.boxplot(data=d)
        plt.xticks(rotation='vertical')
        plt.savefig("plots/boxplot" + str(labelFilename) + ".png")
        plt.close()
        labelFilename+=1"""

def boxplotAll(data, dimension):
    i, j = dimension
    fig, ax = plt.subplots(i, j, figsize=(50, 80))
    
    for position in range(len(data.columns)):
        col = data.columns[position]

        pos_i = position//j
        pos_j = position%j

        ax[pos_i][pos_j].boxplot(data[col])
        ax[pos_i][pos_j].set_title(col)

    plt.savefig('plots/drought_boxplot_ind_all.png')
    plt.close()

def boxplot(data, filename):
    """"""

    fig, ax = plt.subplots()
    sns.boxplot(x=data)
    plt.savefig('plots/'+ filename + ".png")
    plt.close()

def boxplotG(data, filename):
    """"""

    #fig, ax = plt.subplots()
    sns.set(rc={"figure.figsize":(16, 14)})
    sns.boxplot(data=data)
    plt.xticks(rotation='vertical')
    plt.savefig('plots/'+filename + ".png")
    plt.close()

# Groups of attributes to plot together: 
# fips and year
# Interval of values: [0,1) - slope1, slope2, slope3, slope4, slope5, slope6, slope7, slope8, aspectN, aspectE, aspectS, aspectW, aspectUnknown
# Interval of values: [1,10) - WS10M,WS10M_MIN, WS50M_MIN, SQ1, SQ2, SQ3, SQ4, SQ5, SQ6, SQ7, drought
# Interval of values: [0,21) - QV2M,WS10M_MAX, WS10M_RANGE, WS50M, WS50M_MAX, month
# Interval of values: [-20, 50) - T2M, T2MDEW, T2MWET, T2M_MAX, T2M_MIN, T2M_RANGE, TS 
# Interval of values: [-100, 0) - lon
# Interval of values: [-100, 0) - lat
# Interval of values: [0, 800) - elevation
# Interval of values: [0, 102) - WAT_LAND, NVG_LAND, URB_LAND, GRS_LAND, FOR_LAND, CULTRF_LAND, CULTIR_LAND, CULT_LAND, day, PRECTOT

"""dataValuesPerIntervals = [data[["slope1", "slope2", "slope3", "slope4", "slope5", "slope6", "slope7", 
                                "slope8", "aspectN", "aspectE", "aspectS", "aspectW", "aspectUnknown"]], 
                            data[["WS10M","WS10M_MIN", "WS50M_MIN", "SQ1", "SQ2", "SQ3", "SQ4", "SQ5", 
                            "SQ6", "SQ7", "drought"]], 
                            data[["QV2M","WS10M_MAX", "WS10M_RANGE", "WS50M", "WS50M_MAX", "month"]], 
                            data[["T2M", "T2MDEW", "T2MWET", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "TS"]], 
                            data["lon"], data["lat"], data["elevation"], data[["fips", "year"]],
                            data[["WAT_LAND", "NVG_LAND", "URB_LAND", "GRS_LAND", "FOR_LAND", "CULTRF_LAND", 
                            "CULTIR_LAND", "CULT_LAND", "day", "PRECTOT"]]]"""

#boxplots(dataValuesPerIntervals)


"""for col in data.columns:
    if col == 'date':
        continue
    tmp_data = data[col]
    boxplot(tmp_data, "boxplot_drought_" + col)"""

#boxplotG(data, "boxplot_drought_global")

#dataNoDate = data.drop("date", axis=1)
#   boxplotT(dataNoDate, (9,6))

#   3.3. Sparsity:

#     3.3.1. Heatmaps

def heatmap(data):
    heatmap = data.corr()
    f, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(heatmap,
                cmap=sns.color_palette("RdBu_r", 1000),
                vmin=-1,
                vmax=1,
                square=True)
    
    plt.savefig('plots/drought_heatmap01.png')
    plt.close()

#heatmap(data)

#     3.3.2. Scatter plots

def scatterplots(data):
    """ Given the data, the function plot the scatter plots """

    figur = sns.pairplot(data, diag_kind='hist')
    fig = figur.fig
    fig.savefig("plots/scatterplotGeneral.png")

#scatterplots(data)
# it's necessary to define if scatter plots are the best option. considering the number of attributes, maybe it is not. 

#   3.4. Dimensionality:
# 4. prepare the data: 
# 5. learn the model:
# 6. delirevy the model: 
# step 1: change from class to drought
# _____________________________
# ----------- Classification -----------
# Distance matrix

# nested loop algorithm (described in slides provided by the authors from the text-book)
# n = 0.05% do número de entradas , n = 298
# n = 0.05% do número de entradas , n = 298
# r = 574 (not used)
# r = 18,5

countOutliers = 0
#b = []
pos_outliers = []
frac = 0.05*len(data)
for c in range(len(data)):
    euclidean_matrix = euclidean_distances(data.iloc[[c]], data)
    #tmpw = np.sort(euclidean_matrix)
    #a = tmpw[0][0:299]
    #d = (a.sum())/len(a)
    #b.append(d)
    tmp_df = pd.DataFrame(euclidean_matrix)
    tmp_df = tmp_df < 299
    if tmp_df.values.sum() < frac:
        countOutliers+=1
        pos_outliers.append(c)

dataNoOutliers = data.drop(pos_outliers, axis=0)
print(dataNoOutliers)
#e = np.array(b)
#f = (e.sum())/len(e)
#print(f)
print(countOutliers)
# valor de corte de outliers
# método de critério: 
# - calcular a distancia de um ponto aos demais pontos
# - ordenar do maior para o menor e retirar 0,5% dos primeiros maiores, calcular 
# testing the outliers treatment using KNN and Naive Bayes

# KNN
def knn(X_train, X_test, Y_train, Y_test):

    model = KNeighborsClassifier(p=1)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

# Naive Bayes
def nb(X_train, X_test, Y_train, Y_test):

    model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

def temporal_data_split(data, target, train_size=0.8):

    lim = round(len(data)*train_size)
    tmp_data_train = data.loc[0:lim]
    tmp_data_test = data.loc[lim:]
    X_train = tmp_data_train.drop([target],axis=1) 
    Y_train = tmp_data_train.data[target] 
    X_test = tmp_data_test.drop([target],axis=1)
    Y_test = tmp_data_test.data[target]
    return X_train, X_test, Y_train, Y_test

#_____________________________________________________________
# Outliers treatment
# 15 alternatives of scenarios
# Alternative 1



# Alternative 2


#_____________________________________________________________
# Scaling
# Alternative 1
# Standardization
std_scaler = StandardScaler()
data_std = std_scaler.fit_transform(data)

# Alternative 2
# Normalization
norm = Normalizer()
data_norm = norm.fit_transform(data)

# Alternative 3
# No adjustment

#_____________________________________________________________
# Feature Engineering
# Alternative 1



# Alternative 2


#_____________________________________________________________
# Train-Test spliting

