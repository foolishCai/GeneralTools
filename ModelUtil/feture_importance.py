#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 15:04
# @Author  : Cai
# @File    : feture_importance.py
# @Note    : 建模初期，对模型特征值的判断

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt

def Lgb_ImportantFeature(self, sample_num, figsize, max_num_features=20):
    '''
    param sample_num:抽样计算数量
    param figsize:画图的图片大小
    param max_num_features:lgb画图显示top个数
    '''
    # 抽样计算
    data = self.data.sample(n=sample_num, axis=0)
    data = data.drop(self.drop_col, axis=1)
    # 类别型编码，连续型填补缺失
    for i in data.columns:
        if i in self.category_col:
            data[i] = data[i].factorize()[0]
        else:
            data[i] = data[i].fillna(np.nan)
            # 提取解释变量和目标变量
    X = data.drop(self.target, axis=1)
    y = data[self.target]
    # 转换为lgb所需格式数据
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=self.category_col)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, categorical_feature=self.category_col)

    # 参数
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        # 'num_class':3,
        'metric': 'auc',
        'learning_rate': 0.1
        # 'num_leaves':25,
        # 'max_depth':3,
        # 'max_bin':10,
        # 'min_data_in_leaf':8,
        # 'feature_fraction': 0.8,
        # 'bagging_fraction': 1,
        # 'bagging_freq':0,
        # 'lambda_l1': 0,
        # 'lambda_l2': 0,
        # 'min_split_gain': 0
    }

    gbm = lgb.train(params,  # 参数字典
                    lgb_train,  # 训练集
                    num_boost_round=500,  # 迭代次数
                    valid_sets=lgb_eval,  # 验证集
                    verbose_eval=0,  # 验证集评价指标打印，verbose_eval=100则每100轮打印一次
                    early_stopping_rounds=30)  # 早停系数

    # 交叉验证的最优结果
    feature_importance_values = np.zeros(len(X.columns))
    feature_importance_values = np.array(gbm.feature_importance(importance_type='gain'))
    # 输出特征评分
    imp = pd.DataFrame(X.columns.tolist(), columns=['feature'])
    imp['importance'] = feature_importance_values
    imp = imp.sort_values(by='importance', ascending=False)
    imp['normalized_importance'] = imp['importance'] / imp['importance'].sum()
    imp['cumulative_importance'] = np.cumsum(imp['normalized_importance'])
    imp['importance'] = list(map(lambda x: round(x, 2), imp['importance']))
    imp['normalized_importance'] = list(map(lambda x: round(x, 2), imp['normalized_importance']))
    imp['cumulative_importance'] = list(map(lambda x: round(x, 4), imp['cumulative_importance']))
    # 画图
    lgb.plot_importance(gbm, importance_type='gain', max_num_features=max_num_features, precision=0, figsize=figsize)
    plt.title("Feature Importance By Gain")
    plt.savefig('lgb_feature_importance.png', dpi=500)
    plt.show()

    return imp