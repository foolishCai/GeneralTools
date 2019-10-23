#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Date    : 2019/9/23 11:06
# Author  : Cai
# File    : bin_chi2.py
# Note    ：卡方分箱

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class bin_chi2(object):
    def __init__(self, df, target_name, classes_num=None):
        self.df = df
        self.target_name = target_name
        self.allFeatures = self.df.columns.tolist()
        # 连续型变量
        self.categoricalFeatures = []
        # 离散型变量
        self.numericalFeatures = []
        self.WOE_IV_dict = {}
        if classes_num is None:
            self.classes_num = 5
        else:
            self.classes_num = classes_num

    def go_cut(self):
        for var in self.allFeatures:
            if len(set(self.df[var])) > self.classes_num:
                self.numericalFeatures.append(var)
            else:
                self.categoricalFeatures.append(var)

        ##判定类别型变量的单调性,类别型变量中非单调变量
        not_monotone = []
        for var in self.categoricalFeatures:
            # 检查bad rate在箱中的单调性
            if not self.BadRateMonotone(self.df, var, self.target_name):
                not_monotone.append(var)
        print("数值取值小于{}类别型变量{}坏样本率不单调".format(self.classes_num, not_monotone))


        #对其他单调的类别型变量，检查是否有一箱的占比低于5%。 如果有，将该变量进行合并
        small_bin_var = []
        large_bin_var = []
        N = self.df.shape[0]
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                total = self.df.groupby([var])[var].count()
                pcnt = total * 1.0 / N
                if min(pcnt) < 0.05:
                    small_bin_var.append({var: pcnt.to_dict()})  ##分段中有样本量低于5%的分段，需合并
                else:
                    large_bin_var.append(var)

        # 由于有部分箱占了很大比例，故删除，因为样本表现99%都一样，这个变量没有区分度
        for i in range(len(small_bin_var)):
            dict1 = small_bin_var[i]
            var_name = list(dict1.keys())[0]
            var_value = dict1.values()
            var_value1 = list(var_value)[0]
            max_ratio = max(var_value1.values())
            if max_ratio > 0.95:
                self.allFeatures.remove(var_name)

        # 对于small_bin_var中的其他变量，将最小的箱和相邻的箱进行合并并计算WOE
        self.small_bin_var = [i for i in small_bin_var if i in self.allFeatures]
        self.large_bin_var = large_bin_var


        if len(small_bin_var) > 0:
            print("需要合并的特征有：".format(small_bin_var))
            ## for example
            # self.df[new_var_name] = self.df[var].apply(lambda x: self.MergeByCondition(x, ['==0', '>0']))
            # large_bin_var.append(new_var_name)

        # 对于不需要合并、原始箱的bad rate单调的特征，直接计算WOE和IV
        for var in self.large_bin_var:
            self.WOE_IV_dict[var] = self.CalcWOE(self.df, var, 'label')

    ## 判断某变量的坏样本率是否单调
    def BadRateMonotone(self, df, sortByVar, target, special_attribute=[]):
        '''
        :param df: 包含检验坏样本率的变量，和目标变量
        :param sortByVar: 需要检验坏样本率的变量
        :param target: 目标变量，0、1表示好、坏
        :param special_attribute: 不参与检验的特殊值
        :return: 坏样本率单调与否
        '''
        df2 = df.loc[~df[sortByVar].isin(special_attribute)]
        if len(set(df2[sortByVar])) <= 2:
            return True
        regroup = self.BinBadRate(df2, sortByVar, target)[1]
        combined = zip(regroup['total'], regroup['bad'])
        badRate = [x[1] * 1.0 / x[0] for x in combined]
        badRateNotMonotone = [
            badRate[i] < badRate[i + 1] and badRate[i] < badRate[i - 1] or badRate[i] > badRate[i + 1] and badRate[i] >
            badRate[i - 1]
            for i in range(1, len(badRate) - 1)]
        if True in badRateNotMonotone:
            return False
        else:
            return True

    # 计算变量分箱之后各分箱的坏样本率
    @ staticmethod
    def BinBadRate(df, col, target, grantRateIndicator=0):
        '''
        :param df: 需要计算好坏比率的数据集
        :param col: 需要计算好坏比率的特征
        :param target: 好坏标签
        :param grantRateIndicator: 1返回总体的坏样本率，0不返回
        :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
        '''
        # print(df.groupby([col])[target])
        total = df.groupby([col])[target].count()
        # print(total)
        total = pd.DataFrame({'total': total})
        # print(total)
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        # 合并
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        # print(regroup)
        regroup.reset_index(level=0, inplace=True)
        # print(regroup)
        # 计算坏样本率
        regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
        # print(regroup)
        # 生成字典，（变量名取值：坏样本率）
        dicts = dict(zip(regroup[col], regroup['bad_rate']))
        if grantRateIndicator == 0:
            return (dicts, regroup)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        # 总体样本率
        overallRate = B * 1.0 / N
        return (dicts, regroup, overallRate)

    # 合并分箱
    @ staticmethod
    def MergeByCondition(x, condition_list):
        # condition_list是条件列表。满足第几个condition，就输出几
        s = 0
        for condition in condition_list:
            if eval(str(x) + condition):
                return s
            else:
                s += 1
        return s

    @ staticmethod
    # 计算WOE值
    def CalcWOE(df, col, target):
        '''
        :param df: 包含需要计算WOE的变量和目标变量
        :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
        :param target: 目标变量，0、1表示好、坏
        :return: 返回WOE和IV
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
        regroup['WOE'] = regroup.apply(lambda x: round(np.log(x.good_pcnt * 1.0 / x.bad_pcnt), 4), axis=1)
        WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
        for k, v in WOE_dict.items():
            WOE_dict[k] = v['WOE']
        IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
        IV = sum(IV)
        return {"WOE": WOE_dict, 'IV': IV}


    def go_analysis(self):
        # 单变量分析：选取IV高于0.02的变量
        high_IV = [(k, v['IV']) for k, v in self.WOE_IV_dict.items() if v['IV'] >= 0.02]
        high_IV_sorted = sorted(high_IV, key=lambda k: k[1], reverse=True)
        IV_values = [i[1] for i in high_IV_sorted]
        IV_name = [i[0] for i in high_IV_sorted]
        plt.title('High feature IV')
        plt.bar(range(len(IV_values)), IV_values)
        for (var, iv) in high_IV:
            newVar = var + "_WOE"
            self.df[newVar] = self.df[var].map(lambda x: self.WOE_IV_dict[var]['WOE'][x])

        # 多变量分析：比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
        deleted_index = []
        cnt_vars = len(high_IV_sorted)
        for i in range(cnt_vars):
            if i in deleted_index:
                continue
            x1 = high_IV_sorted[i][0] + "_WOE"
            for j in range(cnt_vars):
                if i == j or j in deleted_index:
                    continue
                y1 = high_IV_sorted[j][0] + "_WOE"
                roh = np.corrcoef(self.df[x1], self.df[y1])[0, 1]
                if abs(roh) > 0.7:
                    x1_IV = high_IV_sorted[i][1]
                    y1_IV = high_IV_sorted[j][1]
                    if x1_IV > y1_IV:
                        deleted_index.append(j)
                    else:
                        deleted_index.append(i)

        single_analysis_vars = [high_IV_sorted[i][0] + "_WOE" for i in range(cnt_vars) if i not in deleted_index]

        X = self.df[single_analysis_vars]
        f, ax = plt.subplots(figsize=(10, 8))
        corr = X.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)

        single_analysis_vars.remove('label_WOE')
        multi_analysis = single_analysis_vars
        return multi_analysis