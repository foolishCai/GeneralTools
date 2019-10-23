#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 11:30
@desc:
'''

import pandas as pd
from sklearn import metrics
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_ks_auc(y_pred, y_true):
    fpr, tpr, thresh_roc = metrics.roc_curve(y_true, y_pred)
    ks = abs(fpr - tpr).max()
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    return ks, auc_score

def get_ks_table(y_true, y_pred, gap=None):
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


def get_psi_value(df, train_name, test_name):
    df['psi_value'] = df.apply(lambda x: float("inf") if x[test_name]==0 else float("-inf") if x[train_name]==0 else (x[train_name]-x[test_name])*np.log(x[train_name]-x[test_name]), axis=1)
    return df


# 计算WOE和IV值
def get_woe_iv(df, col, target):
    '''
    : df dataframe
    : col 注意这列已经分过箱了，现在计算每箱的WOE和总的IV
    ：target 目标列 0-1值
    ：return 返回每箱的WOE和总的IV
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    IV_SUM = sum(IV)
    return {'WOE': WOE_dict, 'IV_sum': IV_SUM, 'IV': IV}


# 取得两个变量之间的相关性
def get_corr(df, iv_dict):
    # 检查变量两两间相关系数
    columns = df.columns.tolist()
    id_del = []
    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns)):
            corr2 = df[[columns[i], columns[j]]].corr().iloc[0, 1]
            if abs(corr2) >= 0.7:
                if iv_dict[columns[i]] <= iv_dict[columns[j]]:
                    id_del.append(i)
                else:
                    id_del.append(j)
    # feat_weak_corr内任意两个特征相关性绝对值小于0.7
    feat_weak_corr = [i for i in columns if i not in id_del]
    return feat_weak_corr

def get_vif_value(df):
    col = list(range(df.shape[1]))
    vif = [variance_inflation_factor(df.iloc[:, col].values, ix) for ix in range(len(col))]
    print(max(vif))