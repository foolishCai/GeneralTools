# -*- coding:utf-8 -*-

'''
Created date: 2019-05-17

@author: Cai

note: 逻辑回归模型，评分卡实现
'''

import numpy as np
import pandas as pd
import re

def get_ab(points=600, odds=1, pdo=20):
    """
    逻辑回归函数  p(y=1) = 1/(1+exp(-z))，
        其中 z = beta0 + beta1*x1 + ... + betar*xr = beta*x
    计算得到：z=log(p/(1-p))
    令 odds = p(1-p)
    则 z = log(odds)
       p = odds/(1+odds)
    --------------------------------------------------------------
    评分卡计算方式为
    score = a - b*log(odds)

    这个公式基于两点假设：
    （1) p/(1-p)越大，得分越小，线性关系 points = a-b*log(odds)
     (2) odds比增大一倍时，得分降低pdo分points - PDO = a - b*log(2*odds)
    --------------------------------------------------------------
    根据这两点假设：可以计算得到公式中的啊a,b
    --------------------------------------------------------------
    实际中常用公式：score = 600 - (20/log(2)) * log(p/(1-p)）
    此时，a = 600
         b = 20/log(2)
       pdo = 20
       odds = 1

    也就是好坏比=1时，默认的分数是600分，好坏比（p/(1-p))增加一倍时，分数降低20分
    :param point: 基础得分
    :param odds:  p/(1-p)根据实际情况调整
    :param pdo:  p/(1-p)增加一倍时，降低的分数
    :return: 系数a, b
    """
    # 分数越大，坏人可能性越小
    if pdo>0:
        b=pdo/np.log(2)
    else:
        b = -pdo/np.log(2)

    a = points + b * np.log(odds)
    return a,b




def scorecard(bins, model, xcolumns, points=600, odds=1, pdo=20, basepoints_eq0=False):
    """
    创建评分卡
    :param bins: woe分组，['variable', 'bin', 'woe']
    :param model: （默认为逻辑回归模型）参数list
    :param xcolumns: feature_vars
    :param points: 基础分，默认为600分
    :param op/(1-p) = 1 ,好坏比占比为1:1时，得分为600分
    pdo: 好坏比增加1倍时，降低20分
    basepoints_eq0: 基础分是否为0分
    :return: 评分卡字典

    # example
    card = {}
    for i,model_name in  enumerate(lr_tree.lr_clfs.keys()):
        _bin = woe_data[model_name]['bins']
        _model = lr_tree.lr_clfs[model_name]
        _columns = lr_tree.lr_clf_columns[model_name]
        print(model_name,"截距项系数：",_model.intercept_[0])
        card[model_name] = score_card.scorecard(_bin, _model, _columns,basepoints_eq0=True)
    """

    # 公式系数
    a,b = get_ab(points, odds, pdo)

    # 根据分段的字典形成一个dataframe
    if isinstance(bins, dict):
        bins = pd.DataFrame(bins, ignore_index=True)

    # 把带有_woe结尾的字段名称剔除掉
    xs = [re.sub('_woe$', '', i) for i in xcolumns]

    # 保留模型系数不为0的项
    coef_df = pd.Series(model.coef_[0], index=np.array(xs)).loc[lambda x: x!=0]

    # 输出评分卡
    len_x = len(coef_df)

    # 用户基础分（也是根据系数分，模型所有项都为0的得分)
    basepoints = a - b * model.intercept_[0]
    card = {}
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable': "basepoints", 'bin': np.nan, 'points': 0}, index=np.arange(1))
        for i in coef_df.index:
            bin = bins.loc[bins['variable'] == i, ['variable', 'bin', 'woe']]
            bin[points] = round(-b * bin['woe'] * coef_df[i] + basepoints/len_x)
            card[i] = bin[['variable', 'bin', 'points']]
    else:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':round(basepoints)},index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable'] == i, ['variable', 'bin', 'woe']] \
                .assign(points=lambda x: round(-b * x['woe'] * coef_df[i])) \
                [["variable", "bin", "points"]]
    return card




