#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/30 10:24
@desc: 尝试自己写一下best-ks，大工程啊啊啊啊啊
'''

import numpy as np
from configs import log
from interval import Interval
from ModelUtil.evaluate_util import get_woe_iv
import queue


class KsCutUtil(object):
    def __init__(self, df, target_name, limit_ratio, bins=5, not_need_cut=None, special_dict={}):
        if not_need_cut is not None:
            self.df = df.drop(not_need_cut, axis=1)
        else:
            self.df = df.copy()
        self.target_name = target_name
        self.limit_num = np.floor(len(self.df) * limit_ratio)
        self.bins = bins
        self.cut_point = {}
        self.log = log
        self.global_good = self.df[self.target_name].value_counts()[0]
        self.global_bad = self.df[self.target_name].value_counts()[1]
        self.special_dict = special_dict

    # 取得max_ks, 循环得到cutPoints
    def ks_bin(self, df, feature_name):
        log.info("这是变量{}第{}次分箱".format(feature_name, 1 + len(self.cut_point[feature_name])))
        data_cro = df[self.target_name].groupby([df[feature_name], df[self.target_name]]).count()
        data_cro = data_cro.unstack()
        data_cro.columns = ['good', 'bad']
        data_cro = data_cro.fillna(0)
        data_cro['good_ratio'] = data_cro.good/self.global_good
        data_cro['bad_ratio'] = data_cro.bad/self.global_bad
        data_cro['good_cumsum'] = data_cro.good_ratio.cumsum()
        data_cro['bad_cumsum'] = data_cro.bad_ratio.cumsum()
        data_cro['ks_abs'] = data_cro.apply(lambda x: np.abs(x.good_cumsum - x.bad_cumsum), axis=1)
        ks_index_list = data_cro.sort_values(by='ks_abs', ascending=False).index.tolist()
        for index in ks_index_list:
            flag1_len_data = len(df[df[feature_name] <= index]) > self.limit_num
            flag2_len_data = len(df[df[feature_name] > index]) > self.limit_num
            flag1_label = len(df[df[feature_name] <= index][self.target_name].unique()) > 1
            flag2_label = len(df[df[feature_name] > index][self.target_name].unique()) > 1
            flag1_feature = len(df[df[feature_name] <= index][feature_name].unique()) > 1
            flag2_feature = len(df[df[feature_name] > index][feature_name].unique()) > 1
            if flag1_len_data and flag2_len_data and flag1_label and flag2_label:
                self.cut_point[feature_name].append(index)
                if len(self.cut_point[feature_name]) + 1 >= self.bins:
                    return
                if flag1_feature:
                    self.bins_queue.put((df[df[feature_name] <= index][feature_name].min(), index))
                elif flag2_feature:
                    self.bins_queue.put((index, df[df[feature_name] > index][feature_name].max()))
                if not self.bins_queue.empty():
                    min_value, max_value = self.bins_queue.get()
                    df = df.query('{} > {} and {} <= {}'.format(feature_name, min_value, feature_name, max_value))
                    log.info("分箱的数据范围是：({},{}]".format(min_value, max_value))
                    self.ks_bin(df, feature_name)
                    return
                else:
                    return
            else:
                continue

    # 取得分箱区间
    def cut_bins(self, feature_name):
        cutPoints = self.cut_point[feature_name]
        cutPoints = list(set(cutPoints)) + self.special_dict[feature_name]
        cutPoints.extend([float('-inf'), float('inf')])
        cutPoints.sort()
        bin_list = [Interval(cutPoints[i], cutPoints[i+1], lower_closed=False) for i in range(len(cutPoints)-1)]
        self.df['CutBin_'+feature_name] = self.df[feature_name].map(lambda x: str(bin_list[[x in i for i in bin_list].index(True)]))

    # 计算woe值
    def get_woe_iv(self, feature_name):
        self.bin_reault = {}
        self.bin_reault[feature_name] = get_woe_iv(self.df, 'CutBin_'+feature_name, self.target_name)

    # 用woe值代替原来的连续值
    def change_origin_value(self, feature_name):
        self.df[feature_name] = self.df['CutBin_'+feature_name].map(lambda x: self.bin_reault[feature_name]['WOE'][x]['WOE'])


    def main(self, is_change=True):
        for column in list(self.df.columns):
            if column == self.target_name:
                continue
            self.cut_point[column] = []
            self.bins_queue = queue.Queue()
            df = self.df[~self.df[column].isin(self.special_dict[column])]
            log.info("分箱的数据范围是：[{},{}]".format(df[feature_name].min(), df[feature_name].max()))
            self.ks_bin(df, column)
            self.cut_bins(column)
            self.get_woe_iv(column)
            if is_change:
                self.change_origin_value(column)


import pandas as pd
df = pd.read_csv("/Users/cai/Desktop/pythonProjects/github_FoolishCai/GeneralTools/notImportance/test.txt", sep="|")

feature_name = 'ft_gezhen_multi_loan_score_0'
target_name = 'label'
kcu = KsCutUtil(df[['ft_gezhen_multi_loan_score_0','label']], 'label', limit_ratio=0.05, bins=8, special_dict={'ft_gezhen_multi_loan_score_0': [-1]})
kcu.main(is_change=False)