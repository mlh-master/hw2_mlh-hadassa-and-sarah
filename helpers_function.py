import pandas as pd 
import numpy as np
from pathlib import Path
import random
import warnings

T1D_dataset = pd.read_csv('HW2_data.csv') # load the data

#print(T1D_dataset)

# We go through the list of features (without extra_feature), transform the
# non-numerical values to NaN, then replace the NaN by random values of the
# same column and put the values in a dictionary.
def nan2rand_val(dataset):
    """
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    dict_dataset = {}
    
    lft = list(dataset.columns.values)
    for ft in lft:
        col = dataset[ft].copy()
        #col = pd.to_numeric(col, errors='coerce')
        nan_list = np.array(col.isnull())
        val_list = col[col.notnull()].array
        for i in range(len(nan_list)):
            if nan_list[i] == True:
                col.iloc[i] = np.random.choice(val_list)
        dict_dataset[ft] = col
    # -------------------------------------------------------------------------
    return pd.DataFrame(dict_dataset)

#print(nan2rand_val(T1D_dataset))

def to_one_hot(set):
    warnings.filterwarnings("ignore", category=FutureWarning)
    lft = list(set.columns.values)
    dict_dataset = {}
    for ft in lft:
        col = np.array(set[ft].copy())
        col[col=='Yes'] = 1
        col[col=='Positive'] = 1
        col[col=='Female'] = 1
        col[col=='No'] = 0
        col[col=='Negative'] = 0
        col[col=='Male'] = 0
        dict_dataset[ft] = col
        
    return pd.DataFrame(dict_dataset)
print(T1D_dataset)
print(to_one_hot(T1D_dataset))