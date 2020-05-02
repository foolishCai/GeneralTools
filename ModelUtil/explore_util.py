#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 10:14
@desc:
'''

import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import operator
import math
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from interval import Interval
from scipy.stats import ks_2samp

from BaseUtils.log_util import LogUtil
from ModelUtil.draw_util import format_dataframe

## 特征分析
class OriginExplore(object):
    def __init__(self, df, target_name="y", ids_cols=[], log_name="eda_explore", isChange=True, max_col_null_pct=0.8, max_row_null_pct=0.8,
                 max_null_pct_delta=0.3, time_col="create_date", time_level="m", max_various_values=100,
                 corr_threshold=0.75, vif_threshold=10, psi_threshold=0.2, ks_threshold=0.05):
        self.df = df
        self.df.rename(columns={target_name: "y"}, inplace=True)
        self.ids_cols = []
        self.log = LogUtil(log_name)
        self.max_col_null_pct = max_col_null_pct
        self.max_row_null_pct = max_row_null_pct
        self.max_null_pct_delta = max_null_pct_delta
        self.isChange = isChange
        if max_various_values <= self.df.shape[0]:
            self.max_various_values = max_various_values
        else:
            self.max_various_values = self.df.shape[0]
        self.time_col = time_col
        self.time_level = time_level
        if time_col and time_level:
            from CommonUtils.time_util import get_std_time
            self.df["format_date"] = self.df[self.time_col].map(lambda x: get_std_time(x, level=self.time_level))
            self.ids_cols.append('format_date')
            self.psi_threshold = psi_threshold
            self.ks_threshold = ks_threshold
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold

    # 定义打印对象时打印的字符串
    def __str__(self, print_all=True):
        if print_all:
            self.log.info(self.__dict__)
        else:
            self.log.info(' '.join(('%s' % item for item in self.__dict__.keys())))


    def go_init_explore(self):
        self.check_y()
        if self.time_col and self.time_level:
            self.observe_y_by_time()
        self.observe_all_nulls()
        self.observe_null_by_y()
        self.get_object_cols()
        self.get_single_various_values()
        self.get_null_cols_rows()
        self.output_objects()

    def go_deep_explore(self):
        self.get_corr_explore()
        self.get_vif_explore()
        if self.time_col and self.time_level:
            if (self.df.format_date.unique()) >= 2:
                self.get_psi_ks_explore()

    ## 检查Y值
    def check_y(self):
        data = self.df.copy()
        if len(data.y.unique()) > 2 :
            self.log.info("哇哦！活见鬼的多分类！")
            self.log.info("本函数不适合你，拜拜！")
            sys.exit()
        if not any(data.y.isnull()):
            self.log.info("有{}个样本没有匹配到Y值".format(len(data[data.y.isnull()])))
            self.log.info("全局label1_ratio={}".format(round(data.y.value_counts()[1]/len(data),4)))
            self.log.info("没有匹配到Y值的样本label1_ratio={}".format(round(data[data.y.isnull()].y.value_counts()[1]/len(data[data.y.isnull()]),4)))
            self.df = self.df[~self.df.y.isnull()]
        else:
            self.log.info("所有样本均有Y值")
            self.log.info("但请不要得瑟，数据已经有了，模型，嘿嘿嘿～")
        del data

    ## 检查各个时间单位上的Y分布
    def observe_y_by_time(self):
        data = self.df.copy()
        data = data.y.groupby([data["format_date"]]).agg(["count", "sum"]).sort_values(by="format_date")
        data.rename(columns={"count": "total", "sum":"bad_cnt"}, inplace=True)
        data["bad_pct"] = data.apply(lambda x: round(x.bad_cnt/ x.total, 4), axis=1)
        self.log.info("样本共有{}个时间单位，具体说明如下")
        from ModelUtil.draw_util import get_badpct_by_time
        get_badpct_by_time(data)
        del data

    ## 查看样本总体缺失值情况
    def observe_all_nulls(self):
        data = df.copy()
        data['null_cnt'] = data.isnull().sum(axis=1)
        data['null_pct'] = data.null_cnt / data.shape[0]
        data = data.sort_values('null_pct')

        from ModelUtil.draw_util import get_all_null
        get_all_null(data)
        del data

    ## 分y查看样本缺失情况
    def observe_null_by_y(self):
        data = df.copy()
        null_pct_by_by_label = data.drop('y', axis=1).groupby(data['y']).apply(
            lambda x: (x.isna().sum()) / (x.shape[0])).T.drop_duplicates(0.0).sort_values(0.0)
        null_pct_by_by_label.columns = ['label_0', 'label_1']
        from ModelUtil.draw_util import  get_null_pct_by_label
        get_null_pct_by_label(null_pct_by_by_label)

        null_pct_by_by_label['delta'] = null_pct_by_by_label.apply(lambda x: np.abs(x['label_0'] - x['label_1']), axis=1)
        null_pct_by_by_label = null_pct_by_by_label[null_pct_by_by_label.delta > self.max_null_pct_delta]
        null_pct_by_by_label.sort_values(by=['delta'], ascending=False)
        self.log.info("有{}个特征在好坏样本上缺失值(高于{})差异明显".format(len(null_pct_by_by_label), self.max_null_pct_delta))
        if len(null_pct_by_by_label)>0:
            print(("不同label下的缺失值情况如下\n:{}".format(format_dataframe(null_pct_by_by_label))))
        del data
        del null_pct_by_by_label

    ## 获取所有无法转化的连续变量
    def get_object_cols(self):
        data = self.df.copy()
        init_object_cols = list(data.columns[data.dtypes == 'object'])
        self.real_object_cols = []
        unreal_objects_col = []
        for i, col in enumerate(init_object_cols):
            if col in self.ids_cols:
                continue
            try:
                if self.change_type:
                    self.df[col] = self.df[col].astype(np.float)
                else:
                    unreal_objects_col.append(col)
            except:
                self.real_object_cols.append(col)
        if not self.change_type:
            self.log.info("这些离散变量可以处理:{}".format(unreal_objects_col))
        print("处理以后剩下的离散变量共{}个，分别是：{}\n".format(len(self.real_objects_col), self.real_objects_col))
        del data

    ## 获取唯一值、unique value过多值
    def get_single_various_values(self):
        data = self.df.copy()
        self.single_cols = []
        self.various_cols = []
        self.q0equalq99_col = []
        for col in data.columns:
            if col in self.ids_cols:
                continue
            if len(data[col].head(100).unique()) == 1:
                if len(data[col].unique()) == 1:
                    self.single_cols.append(col)

            if col in self.real_object_cols:
                if len(data[col].head(100).unique()) > self.max_various_values:
                    self.various_cols.append(col)
                elif len(data[col].unique()) > self.max_various_values:
                    self.various_cols.append(col)
            else:
                quantile00, quantile99 = data.quantile([.0, .99])
                if quantile00 == quantile99:
                    self.q0equalq99_col.append(col)
        if self.single_cols:
            self.log.info("有{}个特征为唯一值".format(len(self.single_cols)))
            self.log.info("分别如下:{}".format(self.single_cols))
            if self.isChange:
                self.df = self.df.drop(self.single_cols, axis=1)
        if self.single_cols:
            self.log.info("有{}个离散特征值过多".format(len(self.single_cols)))
            self.log.info("分别如下:{}".format(self.single_cols))
            if self.isChange:
                self.df = self.df.drop(self.various_cols, axis=1)
        if self.q0equalq99_col:
            self.log.info("有{}个连续特征无明显差异".format(len(self.q0equalq99_col)))
            self.log.info("分别如下:{}".format(self.q0equalq99_col))
            if self.isChange:
                self.df = self.df.drop(self.q0equalq99_col, axis=1)
        del data

    ## 特征缺失率情况
    def get_null_cols_rows(self):
        data = self.df.copy()
        self.to_drop_cols = []
        self.to_drop_rows = []

        # 对列的探索
        cols_null_percent = data.isnull().sum() / data.shape[0]
        to_drop_cols = cols_null_percent[cols_null_percent > self.max_col_missing_rate]
        to_drop_cols = dict(to_drop_cols)
        self.to_drop_cols = sorted(to_drop_cols.items(), key=operator.itemgetter(1), reverse=True)

        self.log.info("缺失值比例>{}的有{}个变量".format(self.max_col_missing_rate, len(to_drop_cols)))
        result = {
            'dtype': data.dtypes,
            'data_cnt': data.shape[0],
            'null_cnt': data.isnull().sum(),
            'null_ratio': data.isnull().sum() / data.shape[0]
        }
        result = pd.DataFrame(result, columns=['dtype', 'data_cnt', 'null_cnt', 'null_ratio'])
        result = result[result.null_ratio > self.max_col_missing_rate]
        result = result.sort_values(by=['null_ratio'], ascending=False)
        print("每列缺失值情况如下\n:{}".format(format_dataframe(result)))
        if self.isChange:
            self.df = self.df.drop([i[0] for i in self.to_drop_cols], axis=1)

        # 对行的探索
        rows_null_percent = data.isnull().sum(axis=1) / data.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > self.max_row_missing_rate]
        to_drop_rows = to_drop_rows.to_dict()
        self.to_drop_rows = sorted(to_drop_rows.items(), key=operator.itemgetter(1), reverse=True)
        self.log.info("缺失值比例>{}的有{}行".format(self.max_row_null_pct, len(to_drop_rows)))
        data = data[data.index.isin([i[0] for i in to_drop_rows])]
        self.log.info("全局label1的占比为:{}".format(round(self.df.y.value_counts()[1]/len(self.df),4)))
        self.log.info("特征饱和度低于{}的样本label1的占比为:{}".format(round(data.y.value_counts()[1] / len(data), 4)))
        del data
        del result
        del cols_null_percent
        del rows_null_percent

    ## 连续特征相关性
    def get_corr_explore(self):
        data = self.df.drop(self.real_object_cols + self.ids_cols, axis=1)
        cols = list(data.columns)
        corrlist = [(col1, col2, data[[col1, col2]].corr().iloc[0, 1]) for x, y in zip(cols, cols[1:])]
        high_corr_df = pd.DataFrame(data={"col1": [i[0] for i in corrlist],
                                           "col2": [i[1] for i in corrlist],
                                           "corr_value": [i[2] for i in corrlist]})

        self.log.info(">>>>>>>>>>>>>>>>>>>>> 相关系数高于0.75 <<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(format_dataframe(high_corr_df[np.abs(high_corr_df.corr_value) > 0.75]))
        self.log.info("余下特征相关系数如下：\n")
        cols = [i for i in data.columns if i not in high_corr_df.col1 and i not in high_corr_df.col2]
        data = data[cols]
        from ModelUtil.draw_util import get_correlation
        get_correlation(data)
        del data

    ## 连续特征VIF
    def get_vif_explore(self):
        data = self.df.drop(self.real_object_cols, axis=1)
        if data.isnull().any().max() == 1:
            self.log.info("该样本还包含null值，暂用-1填充")
            data = data.fillna(-1)
        num_cols = list(range(self.df.shape[1]))
        vif_df = pd.DataFrame()
        vif_df['vif_value'] = [variance_inflation_factor(self.df.iloc[:, num_cols].values, ix) for ix in range(len(num_cols))]
        vif_df['feature'] = self.df.columns
        vif_df = vif_df[vif_df.vif_value >= self.vif_threshold]
        vif_df = self.vif_df.sort_values(by='vif_value', ascending=False)
        self.log.info("这是高vif的相关说明\n")
        print("max(VIF) = {}".format(max(vif_df.vif_value)))
        del data

    ## 离散变量的psi检验, 连续变量的ks检验
    def get_psi_ks_explore(self):
        date_cols = sorted(list(self.df.format_date.unique()))

        psi_df = pd.DataFrame(columns=date_cols[1:])
        data = self.df[self.real_object_cols + ["format_date"]]
        for col in list(data.columns):
            data[col] = data[col].astype(str)
            ratio_list = list((data[col].groupby([data.format_date]).agg(["count"]) / data.shape[0])["count"])
            ratio_list = [i if i > 0 else 0.000001 for i in ratio_list]
            psi_values = [(y - x) * np.log(y / x) for x, y in zip(ratio_list, ratio_list[1:])]
            psi_df = pd.concat([psi_df, pd.DataFrame(data=psi_values, columns=[col], index=date_cols[1:]).T])
        psi_df["max_value"] = psi_df.max(axis=1)

        ks_df = pd.DataFrame(columns=date_cols[1:])
        data = self.df[[i for i in list(self.df.columns) if i not in self.real_object_cols and i not in self.ids_cols] + ["format_date"]]
        for col in list(data.columns):
            ks_values = [ks_2samp(list(data[data.format_date == x][col]), list(data[data.format_date == y][col]))[1] for x,y in zip(date_cols, date_cols[1:])]
            ks_df = pd.concat([ks_df, pd.DataFrame(data=ks_values, columns=date_cols[1:], index=[col]).T])
        ks_df["max_value"] = ks_df.max(axis=1) <= self.ks_threshold

        psi_unstable = list(psi_df[psi_df.max_value>=self.psi_threshold].index)
        if psi_unstable:
            self.log.info("该样本在psi检测上有{}个特征不稳定，分别是{}".format(len(psi_unstable), psi_unstable))
            print(format_dataframe(psi_df[psi_df.max_value>=self.psi_threshold]))

        ks_unstable = list(ks_df[ks_df.max_value>=self.ks_df].index)
        if ks_unstable:
            self.log.info("该样本在ks检测上有{}个特征不稳定，分别是{}".format(len(ks_unstable), ks_unstable))
            print(format_dataframe(ks_unstable[ks_unstable.max_value >= self.ks_threshold]))


    ## output show
    def output_objects(self):
        self.log.info("本次EDA，可供参考的数据信息仍有以下\n")
        self.log.info("无法转化为连续(float)的特征: {}.real_object_cols".format(xx.__class__.__name__))
        self.log.info("有唯一值的特征: {}.single_cols".format(xx.__class__.__name__))
        self.log.info("value值过多的特征: {}.various_cols".format(xx.__class__.__name__))
        self.log.info("分布无明显变化的特征: {}.q0equalq99_col".format(xx.__class__.__name__))
        self.log.info("缺失率过高的特征: {}.to_drop_cols".format(xx.__class__.__name__))
        self.log.info("缺失率过高的行: {}.to_drop_rows".format(xx.__class__.__name__))