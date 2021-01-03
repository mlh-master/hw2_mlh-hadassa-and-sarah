from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
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

def tune_LinReg(kf, X, y, K=5):
    warnings.filterwarnings("ignore", category=UserWarning)
    print("here")
    validation_dict = []
    C = np.logspace(-1,3,5)
    penalty = ['l1','l2','elasticnet']
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, l1_ratio=0.5, multi_class='ovr')
            loss_val_vec = np.zeros(K)
            k = 0 
            curr_auc = []
            for train_idx, val_idx in kf.split(X, y.iloc[:,1]):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                logreg.fit(x_train,y_train.iloc[:,1])
                y_pred = logreg.decision_function(x_val)
                score = roc_auc_score(y_val.iloc[:,1], y_pred)
                k = k + 1
                curr_auc.append(score)
            elem_dict = {"C": c,
                         "penalty": p,
                         "auc_score": np.median(curr_auc)}
            validation_dict.append(elem_dict)
    list_scores = [elem_dict['auc_score'] for elem_dict in validation_dict]
    return validation_dict[np.argmax(list_scores)]