# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 模型评估各个指标，部分画图
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve


def get_score_table(y_true, y_pred, gap=None):
    _result = pd.DataFrame(columns=['y_true', 'y_pred'])
    _result['y_true'] = y_true
    _result['y_pred'] = y_pred
    if gap is None:
        gap = 0.01

    score_df = pd.DataFrame(columns=['score_theshold', 'total', 'bad_cnt', 'good_cnt', 'accum_bad_cnt',
                                     'accum_total_ratio', 'accum_bad_ratio', 'TPR', 'FPR'])
    total_bad = _result.y_true.sum()
    total_good = _result.y_true.count() - _result.y_true.sum()
    for i in range(100, 0, -int(gap*100)):
        tmp_dict = {}
        min_ = round(i / 100, 2)
        max_ = round(min_ + gap, 2)
        index_name = '[' + str(min_) + ',' + str(max_) + ')'
        tmp_dict['score_theshold'] = str(min_)
        tmp_dict['total'] = len(_result[(_result.y_pred < max_) & (_result.y_pred >= min_)])
        tmp_dict['bad_cnt'] = len(_result[(_result.y_pred < max_) & (_result.y_pred >= min_) & (_result.y_true == 1)])
        tmp_dict['good_cnt'] = len(_result[(_result.y_pred < max_) & (_result.y_pred >= min_) & (_result.y_true == 0)])
        tmp_dict['accum_bad_cnt'] = len(_result[(_result.y_pred >= min_) & (_result.y_true == 1)])
        if len(_result[(_result.y_pred >= min_)]) == 0:
            continue
        tmp_dict['accum_bad_ratio'] = round(tmp_dict['accum_bad_cnt'] / len(_result[_result.y_pred >= min_]), 4)
        tmp_dict['accum_total_ratio'] = round(len(_result[(_result.y_pred >= min_)]) / len(_result), 4)
        tmp_dict['TPR'] = round(len(_result[(_result.y_pred >= min_) & (_result.y_true == 1)]) / total_bad, 4)
        tmp_dict['FPR'] = round(len(_result[(_result.y_pred >= min_) & (_result.y_true == 0)]) / total_good, 4)

        score_df = pd.concat([score_df, pd.DataFrame(tmp_dict, index=[index_name])], axis=0)
    score_df = score_df[
        ['score_theshold', 'total', 'bad_cnt', 'good_cnt', 'accum_bad_cnt','accum_total_ratio', 'accum_bad_ratio', 'TPR', 'FPR']]
    score_df['score_theshold'] = score_df.score_theshold.astype(float)
    score_df = score_df[score_df.total>0]
    return score_df


def get_ks_auc(y_pred, y_true):
    fpr, tpr, thresh_roc = roc_curve(y_true, y_pred)
    ks = abs(fpr - tpr).max()
    auc_score = roc_auc_score(y_true, y_pred)
    return ks, auc_score


def get_lift_table(y_true, y_pred, groupNum=None):
    if groupNum is None:
        groupNum = 10

    N = len(y_true)
    tags = np.linspace(-1, N, groupNum + 1, endpoint=True).astype(int)
    labels = np.arange(1, groupNum + 1)
    df = pd.DataFrame(columns=['prob', 'target'])
    df['prob'] = y_pred
    df['target'] = y_true

    pos = len(df[df['target'] == 1])
    pos_rate = pos / N

    df.sort_values(by=['prob'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['bin'] = pd.cut(df.index.values, bins=tags, labels=labels)

    df_agg = df.groupby(['bin'])['target'].agg(['count', 'sum']).reset_index()
    df_score = df.groupby(['bin'])['prob'].agg(['min', 'max']).reset_index()
    union_df = pd.merge(df_agg, df_score, how='inner', on='bin')
    union_df['score_range'] = union_df.apply(lambda x: str(round(x['min'], 4)) + "~" + str(round(x['max'], 4)), axis=1)
    union_df['top_range'] = union_df.bin.map(
        lambda x: str(100 / groupNum * (x - 1)) + '%~' + str(100 / groupNum * (x)) + '%')

    union_df = union_df.rename(columns={'count': 'total', 'sum': 'bad_cnt'})
    union_df['bad_ratio'] = union_df['bad_cnt'] / union_df['total']
    union_df['lift'] = union_df['bad_ratio'].map(lambda x: round(x / pos_rate, 4))
    union_df['bad_cumsum_ratio'] = union_df['bad_cnt'].cumsum() / pos
    union_df['bad_cumsum_ratio'] = union_df['bad_cumsum_ratio'].map(lambda x: round(x, 4))
    union_df = union_df[['top_range', 'score_range', 'total', 'bad_cnt', 'bad_ratio', 'bad_cumsum_ratio', 'lift']]

    return union_df
