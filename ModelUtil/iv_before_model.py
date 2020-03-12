#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 15:47
# @Author  : Cai
# @File    : get_iv.py
# @Note    : 针对客户各户给过来的样本计算IV；有两种情况；有target值和无target值

import pandas as pd
from configs import log
import woe.feature_process as fp
import math
import toad
from ModelUtil.draw_util import get_iv_plt




class GetIv(object):
    def __init__(self, df, target_name):
        self.log = log
        self.config_df = pd.read_csv("/Users/cai/Desktop/pythonProjects/gitlab_yzx/just-for-cai/feature_configs/import_miss_763.txt", sep=",")
        self.discrete_vars = self.config_df[self.config_df.if_continuous == 0].feature.unique()
        self.continuous_vars = self.config_df[self.config_df.if_continuous == 1].feature.unique()

        df.rename(columns={target_name: 'target'}, inplace=True)
        self.df = df

    def get_woe(self):
        data = self.df.copy()  # 用于存储所有数据的woe值
        self.civ_list = []
        n_positive = sum(self.df['target'])
        n_negtive = len(self.df) - n_positive

        for col in data.columns:
            if col not in self.discrete_vars and col not in self.continuous_vars:
                continue
            else:
                try:
                    if col in self.continuous_vars:
                        data[col] = data[col].astype(float)
                        civ = fp.proc_woe_continuous(data[~data[col].isnull()], col, n_positive, n_negtive, 0.05 * len(data),
                                                 alpha=0.05)
                    elif col in self.discrete_vars:
                        civ = fp.proc_woe_discrete(data[~data[col].isnull()], col, n_positive, n_negtive, 0.05 * len(data),
                                               alpha=0.05)
                    self.civ_list.append(civ)
                    self.df.loc[~(self.df[col].isnull()), col] = fp.woe_trans(data[~data[col].isnull()][col], civ)
                except:
                    self.log.info("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>{} cannot be caculated!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(col))

    def get_data2tag(self):
        cols = self.discrete_vars.tolist() + self.continuous_vars.tolist() + ['target']
        cols = [i for i in cols if i in self.df.columns]
        data = self.df[cols]
        data = data.fillna('blank')
        self.bin_df = pd.DataFrame(columns=['feature', 'tag', 'LabelCnt_0', 'LabelCnt_1'])
        for var in data.columns:
            data[var] = data[var].astype(str)
            if var == 'target':
                continue
            else:
                tagList = self.df[var].unique().tolist()
                self.log.info("The {} tagList is {}\n".format(var, tagList))
                label0List = []
                label1List = []
                for tag in tagList:
                    label0List.append(len(data.query("target==0 and {}=='{}'".format(var, tag))))
                    label1List.append(len(data.query("target==1 and {}=='{}'".format(var, tag))))
                tmp_df = pd.DataFrame(data={"feature": [var] * len(tagList),
                                            "tag": tagList,
                                            "LabelCnt_0": label0List,
                                            "LabelCnt_1": label1List}, columns=['feature', 'tag', 'LabelCnt_0', 'LabelCnt_1'])
                self.bin_df = pd.concat([self.bin_df, tmp_df])
        self.bin_df = self.bin_df[(self.bin_df.LabelCnt_0 > 0) | (self.bin_df.LabelCnt_1 > 0)]

    def get_woe_iv(self):
        self.bin_df['LabelRatio_0'] = self.bin_df.LabelCnt_0.map(lambda x: round(x / sum(self.df.target == 0), 4))
        self.bin_df['LabelRatio_1'] = self.bin_df.LabelCnt_1.map(lambda x: round(x / sum(self.df.target == 1), 4))

        self.bin_df['woe'] = self.bin_df.apply(
            lambda x: float("-inf") if x["LabelRatio_1"] == 0 else float("inf") if x["LabelRatio_0"] == 0 else round(
                math.log(x["LabelRatio_1"] / x["LabelRatio_0"]), 4), axis=1)
        self.bin_df["iv"] = self.bin_df.apply(lambda x: (x.LabelRatio_1 - x.LabelRatio_0) * x.woe, axis=1)
        trand_df = self.config_df[['feature', 'chinese_name']].drop_duplicates()
        self.bin_df = pd.merge(self.bin_df, trand_df, how='left', on='feature')
        self.bin_df = pd.concat(
            [self.bin_df[~self.bin_df.iv.isin(["inf", "-inf"])].sort_values(by='iv', ascending=False),
             self.bin_df[self.bin_df.iv.isin(["inf", "-inf"])]])



class GetIvMap(object):
    def __init__(self, df, target_name, max_iv_ratio=None):
        self.log = log
        df.rename(columns={target_name: 'target'}, inplace=True)
        self.df = df
        if max_iv_ratio is None:
            self.max_iv_ratio = 0.2
        else:
            self.max_iv_ratio = max_iv_ratio

    def get_iv_map(self, is_show=False):
        iv_map = {col: toad.IV(self.df[col], self.df['target']) for col in self.df.columns if col!='target'}
        iv_map = dict(sorted(iv_map.items(), key=lambda kv: kv[1], reverse=True))
        iv_map = {k: v for k, v in iv_map.items() if v > self.max_iv_ratio}
        self.log.info("超过阈值={}的特征有{}个".format(self.max_iv_ratio, len(iv_map)))
        if is_show:
            get_iv_plt(iv_map)