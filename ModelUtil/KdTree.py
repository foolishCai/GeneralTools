#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 16:43
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : KdTree.py
# @Note    :


import sys
sys.path.append("..")
from BaseUtils import log

from sklearn.neighbors import KDTree, BallTree
import numpy as np
import datetime
import operator
import pickle

class makeTree(object):
    def __init__(self, df, file_name=None, max_col_null_pct=0.8, max_row_null_pct=0.8):
        self.log = log
        if file_name:
            self.filename = file_name
        else:
            self.filename = "kdTree_" + datetime.datetime.today().strftime("%Y%m%d")

        self.df = df
        self.max_col_null_pct = max_col_null_pct
        self.max_row_null_pct = max_row_null_pct

    # 定义打印对象时打印的字符串
    def __str__(self, print_all=True):
        if print_all:
            self.log.info(self.__dict__)
        else:
            self.log.info(' '.join(('%s' % item for item in self.__dict__.keys())))

    def explore_null_value(self):

        # 对行的探索
        data = self.df.copy()
        rows_null_percent = data.isnull().sum(axis=1) / data.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > self.max_row_null_pct]
        to_drop_rows = to_drop_rows.to_dict()
        self.to_drop_rows = sorted(to_drop_rows.items(), key=operator.itemgetter(1), reverse=True)
        self.log.info("缺失值比例>{}的有{}行".format(self.max_row_null_pct, len(to_drop_rows)))
        data = data[data.index.isin(list(to_drop_rows.keys()))]
        self.log.info(
                "特征饱和度低于{}的样本占比为:{}".format(self.max_row_null_pct, round(len(data) / len(self.df), 4)))
        self.log.info("*" * 50)

        # 对列的探索
        data = self.df.copy()
        cols_null_percent = data.isnull().sum() / data.shape[0]
        to_drop_cols = cols_null_percent[cols_null_percent > self.max_col_null_pct]
        to_drop_cols = dict(to_drop_cols)
        to_drop_cols = sorted(to_drop_cols.items(), key=operator.itemgetter(1), reverse=True)
        self.log.info("缺失值比例>{}的有{}个变量".format(self.max_col_null_pct, len(to_drop_cols)))

        self.df = self.df[~self.df.index.isin(to_drop_rows.keys())]
        self.df = self.df.drop([i[0] for i in to_drop_cols], axis=1)

    def explore_feature(self):
        data = self.df.copy()
        to_drop_cols = []
        self.standard_dict = {}
        for col in list(data.columns):
            try:
                data[col] = data[col].astype(float)
            except:
                data = data.drop([col], axis=1)
                self.log.info("The col={} is not serial col!".format(col))

            q5, q95 = np.percentile(data[~data[col].isnull()][col].tolist(), 5), np.percentile(
                data[~data[col].isnull()][col].tolist(), 95)
            m, s = (np.mean(data[~data[col].isnull()][col].tolist()), np.std(data[~data[col].isnull()][col].tolist()))
            if q5 == q95:
                self.log.info("This col={}, q5 == q95".format(col))
                to_drop_cols.append(col)
            elif round(m / s, 4) <= 0.15 or s == 0:
                self.log.info("This col={} cv is not standard".format(col))
            else:
                self.standard_dict[col] = (m, s)

        self._save_file(self.standard_dict, self.filename + ".json")
        self.df = self.df[list(self.standard_dict.keys())]

    def get_standard_data(self):
        data_train_df = self.df[[]]
        for col in list(self.standard_dict.keys()):
            data_train_df[col] = self.df[col].astype(float)
            data_train_df.loc[data_train_df[~data_train_df[col].isnull()].index, col] = data_train_df[col].map(lambda x: round((x - self.standard_dict[col][0])/self.standard_dict[col][1], 2))
            data_train_df[col] = data_train_df[col].fillna(self.standard_dict[col][0])
        return data_train_df


    def get_tree(self, df):
        if self.df.shape[1] > 20:
            self.log.info("特征太多，适用ball_tree进行构造")
            self.tree = BallTree(df)
            self._save_file(self.tree, self.filename+".pkl")
        else:
            self.log.info("特征适量，适用kd_tree进行构造")
            self.tree = KDTree(df)
            self._save_file(self.tree, self.filename + ".pkl")
        self.log.info("The tree_model has been finished")

    # def get_predict(self, ):

    def _save_file(self, model, file_name):
        fw = open(file_name, 'wb')
        pickle.dump(model, fw, -1)
        fw.close()

    def main(self):
        self.explore_null_value()
        self.explore_feature()
        data_train_df = self.get_standard_data()
        self.get_tree(data_train_df)