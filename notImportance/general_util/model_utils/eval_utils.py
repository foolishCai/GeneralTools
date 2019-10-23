# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 模型评估各个指标，部分画图
'''

import numpy as np
from sklearn import metrics
import pandas as pd
from collections import Counter
from general_util.config import log
from sklearn.metrics import roc_auc_score,roc_curve

class EvalUtils(object):
    def __init__(self, df, pred_col, true_col):
        self.df = df[[pred_col,true_col]]
        self.df = self.df.rename(columns={pred_col:"y_prob", true_col:"y_true"})
        self.log = log
        if len(Counter(self.df.y_prob.unique())) > 2:
            self.log.info('预测结果不是二分类！！！！')

    def _get_pred(self, threshold):
        self.df['y_pred'] = np.where(self.df.y_prob<=threshold, 0, 1)

    def get_summary_report(self):
        accuracy = metrics.accuracy_score(self.df.y_true, self.df.y_pred)
        prediction, recall, fscore, support = metrics.precision_recall_fscore_support(self.df.y_true, self.df.y_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(self.df.y_true, self.df.y_pred).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(self.df.y_true, self.df.y_prob)
        auc = metrics.auc(fpr, tpr)
        ks_value = abs(fpr - tpr).max()

        summary_report = {
            'accuracy': accuracy,
            'prediction': prediction,
            'recall': recall,
            'fscore': fscore,
            'no_occurance': support,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'auc': auc, 'ks': ks_value
        }
        return summary_report

    def get_confusion_matrix(self):
        cm = metrics.confusion_matrix(self.df.y_true, self.df.y_pred)
        return cm


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
    for i in range(100, -1, -int(gap*100)):
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
        tmp_dict['accum_bad_ratio'] = round(tmp_dict['accum_bad_cnt'] / len(_result[_result.y_true == 1]), 4)
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

