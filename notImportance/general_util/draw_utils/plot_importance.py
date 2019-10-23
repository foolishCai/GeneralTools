# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''

import matplotlib.pyplot as plt


def feat_imp(df, model, top=30):
    """TOP重要变量绘图"""
    feature_nums = len(df.columns)
    if feature_nums < top:
        top = feature_nums
    d = dict(zip(df.columns, model.feature_importances_))
    feature_importance_list = [[k, v] for v, k in sorted(((v, k) for k, v in d.items()), reverse=True)]
    ss = sorted(d, key=d.get, reverse=True)
    top_names = ss[0:top]

    plt.figure(figsize=(8, 6))
    plt.title("Feature importances")
    plt.barh(range(top), [d[i] for i in top_names], color="b", align="center")
    plt.ylim(-1, top)
    plt.yticks(range(top), top_names)
    return feature_importance_list



def feat_imp_lr(varibles, importance, top=30):
    """TOP重要变量绘图"""
    feature_nums = len(varibles)
    if feature_nums < top:
        top = feature_nums
    d = dict(zip(varibles, importance))
    feature_importance_list = sorted(d.items(), key=lambda item: abs(item[1]), reverse=True)
    top_names = [i[0] for i in feature_importance_list][: top]

    plt.figure(figsize=(8, 6))
    plt.title("Feature importances")
    plt.barh(range(top), [d[i] for i in top_names], color="b", align="center")
    plt.ylim(-1, top)
    plt.xlim(min(importance), max(importance))
    plt.yticks(range(top), top_names)
    return feature_importance_list