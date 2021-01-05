from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
import pandas as pd 
import numpy as np
from pathlib import Path
import random
import warnings

T1D_dataset = pd.read_csv('HW2_data.csv') # load the data


# Aims to replace nans with random values
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

# Aims to tune the best parameters for logistic regression
def tune_LogReg(kf, X, y, K=5):
    warnings.filterwarnings("ignore", category=UserWarning)
    validation_dict = []
    C = np.logspace(-1,3,5)
    penalty = ['l1','l2','elasticnet']
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, l1_ratio=0.5, multi_class='ovr')
            loss_val_vec = np.zeros(K)
            k = 0 
            curr_auc = []
            #.iloc[:]
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                logreg.fit(x_train,y_train)
                y_pred = logreg.decision_function(x_val)
                score = roc_auc_score(y_val, y_pred)
                k = k + 1
                curr_auc.append(score)
            elem_dict = {"C": c,
                         "penalty": p,
                         "auc_score": np.median(curr_auc)}
            validation_dict.append(elem_dict)
    list_scores = [elem_dict['auc_score'] for elem_dict in validation_dict]
    return validation_dict[np.argmax(list_scores)]

# Aims to tune the best parameters for random forest
def tune_RandForest(kf, X, y, K=5):
    warnings.filterwarnings("ignore", category=UserWarning)
    np.seterr(divide='ignore', invalid='ignore')
    validation_dict = []
    
    nb_trees = [1,10,100,200]
    crits=['gini','entropy'] #criterion
    bool_oobs = [True, False] #oob_score
    sqrt_nb_ft = int(np.sqrt(X.shape[1]))
    max_ft = range(2, sqrt_nb_ft) #Max features
    counter = 0
    # Going through each combination of parameters and getting the median of the folds' f1 scores
    for nb_tr in nb_trees:
        for crit in crits:
            for bool_oob in bool_oobs:
                for nb_ft in max_ft:
                    clf = RandomForestClassifier(class_weight='balanced', n_estimators=int(nb_tr), criterion=crit, max_features=nb_ft, oob_score=bool_oob)
                    curr_auc = []
                    for train_idx, val_idx in kf.split(X, y):
                        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        clf.fit(x_train,y_train)
                        y_pred = clf.predict_proba(x_val)
                        score = roc_auc_score(y_val, y_pred[:,1])
                        curr_auc.append(score)
                    elem_dict = {"Nb_trees":nb_tr,
                                 "Criterion":crit,
                                 "Max_features":nb_ft,
                                 "oob_score":bool_oob,
                                 "auc_score": np.median(curr_auc)}
                    validation_dict.append(elem_dict)
    list_scores = [elem_dict['auc_score'] for elem_dict in validation_dict]
    return validation_dict[np.argmax(list_scores)]

# Aims to reports different performance score for a given classifier on a given set
def report_performance(clf, X, y, type_decision='decision_function'):
    y_pred = clf.predict(X)
    f1_sc = f1_score(y, y_pred)
    acc_sc = accuracy_score(y, y_pred)
    if type_decision=='decision_function':
        y_pred_prob = clf.decision_function(X)
    else:
        y_pred_prob = clf.predict_proba(X)[:,1]
    auc_sc = roc_auc_score(y, y_pred_prob)
    log_loss_sc = log_loss(y, y_pred_prob)
    
    return [f1_sc, acc_sc, auc_sc, log_loss_sc]