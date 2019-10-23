import os, psutil
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import random
def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('memory GB:', memoryUse)
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
def diff(lst1,lst2):
    return list(set(lst1) - set(lst2))
def unioun(lst1,lst2):
    return list(set(lst1) + set(lst2))


def lgb_importance(traindata):
    X = traindata.drop(['ID', 'TARGET'], axis=1)
    features = list(X.columns)
    X = X.values
    y = traindata['TARGET'].values
    kfold = 5
    nrounds=2000
    ratio = traindata['TARGET'].value_counts()[0]/traindata['TARGET'].value_counts()[1]
    params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500,
              'scale_pos_weight':ratio}
    result = []
    skf = StratifiedKFold(n_splits=kfold, random_state=random.randint(0,99999))
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train,feature_name = features), nrounds, 
                      lgb.Dataset(X_eval, label=y_eval), verbose_eval=500, 
                              early_stopping_rounds=200)
        result.append(lgb_model)
    return result