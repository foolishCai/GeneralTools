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


class KsCutUtil(object):
    def __init__(self, df, target_name, limit_ratio, bins=5):
        self.df = df.copy()
        self.target_name = target_name
        self.limit_num = np.floor(len(self.df) * limit_ratio)
        self.bins = bins
        self.cut_point = {}
        self.log = log

    # 取得max_ks, 循环得到cutPoints
    def ks_bin(self, df, feature_name):
        total_good = df[self.target_name].value_counts()[0]
        total_bad = df[self.target_name].value_counts()[1]
        if total_bad>0 and total_good>0:
            log.info("这是变量{}第几次分箱：{}".format(feature_name, 1 + len(self.cut_point[feature_name])))
            log.info("本次分箱原始好坏样本比为：{} vs {}".format(total_good, total_bad))
        else:
            log.info("这是变量{}第几次分箱：{}".format(feature_name, 1 + len(self.cut_point[feature_name])))
            log.info("本次分箱失败，y标签一致")
            return
        data_cro = df[self.target_name].groupby([df[feature_name], df[self.target_name]]).count()
        data_cro = data_cro.unstack()
        data_cro.columns = ['good', 'bad']
        data_cro = data_cro.fillna(0)
        data_cro['good_ratio'] = data_cro.good/total_good
        data_cro['bad_ratio'] = data_cro.bad/total_bad
        data_cro['good_cumsum'] = data_cro.good_ratio.cumsum()
        data_cro['bad_cumsum'] = data_cro.bad_ratio.cumsum()
        data_cro['ks_abs'] = data_cro.apply(lambda x: np.abs(x.good_cumsum - x.bad_cumsum), axis=1)
        ks_index_list = data_cro.sort_values(by='ks_abs', ascending=False).index.tolist()
        for index in ks_index_list:
            data_1 = df[df[feature_name] <= index]
            data_2 = df[df[feature_name] > index]
            if len(data_1) >= self.limit_num and len(data_2) >= self.limit_num:
                self.cut_point[feature_name].append(index)
                if len(self.cut_point[feature_name]) == self.bins - 1:
                    return
                else:
                    if len(data_1) > len(data_2) and len(data_1[self.target_name].unique()) == 2 and len(data_1[feature_name].unique())>1:
                        self.ks_bin(df=data_1, feature_name=feature_name)
                        return
                    elif len(data_2) > len(data_1) and len(data_2[self.target_name].unique()) == 2 and len(data_2[feature_name].unique())>1:
                        self.ks_bin(df=data_2, feature_name=feature_name)
                        return
            elif len(data_1) >= self.limit_num and len(data_1[self.target_name].unique()) == 2 and len(data_1[feature_name].unique())>1:
                self.ks_bin(df=data_1, feature_name=feature_name)
                return
            elif len(data_2) >= self.limit_num and len(data_2[self.target_name].unique()) == 2 and len(data_2[feature_name].unique())>1:
                self.ks_bin(df=data_2, feature_name=feature_name)
                return
            else:
                return

    # 取得分箱区间
    def cut_bins(self, feature_name):
        cutPoints = self.cut_point[feature_name]
        cutPoints = list(set(cutPoints))
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


    def main(self, not_need_cut=[], is_change=True):
        for column in list(self.df.columns):
            if column not in not_need_cut:
                data = self.df.copy()
                self.cut_point[column] = []
                self.ks_bin(data, column)
                self.cut_bins(column)
                self.get_woe_iv(column)
                if is_change:
                    self.change_origin_value()

