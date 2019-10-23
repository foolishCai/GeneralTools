# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 查找最优参数
'''

import pandas as pd
from sklearn.model_selection import GridSearchCV

def bin_classify(clf, features, target, params=None, score=None):
    '''在一个分类器上寻找最优参数'''

    global Xtrain, Xtest, y_train, y_test
    try:
        Xtrain = Xtrain[features]
        Xtest = Xtest[features]
        print(Xtrain.shape, y_train.shape)
        y_train = y_train[target]
        y_test = y_test[target]
    except:
        raise Exception('split data 出错')

    # 网格搜索最佳参数
    grid_search = GridSearchCV(estimator=clf,
                               param_grid=params,
                               cv=5,
                               scoring=score,
                               n_jobs=-1)
    grid_search.fit(Xtrain, y_train)
    best_model = grid_search.best_estimator_

    # 预测值
    y_pred = grid_search.predict(Xtest)

    if hasattr(grid_search, 'predict_proba'):
        y_score = grid_search.predict_proba(Xtest)[:, 1]
    elif hasattr(grid_search, 'decision_function'):
        y_score = grid_search.decision_function(Xtest)
    else:
        y_score = y_pred

    predictions = {'y_pred': y_pred, 'y_score': y_score}
    df_predictions = pd.DataFrame.from_dict(predictions)

    return best_model, df_predictions

