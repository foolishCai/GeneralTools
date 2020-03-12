#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 10:14
@desc:
'''

from configs import log
from collections import Counter
import pandas as pd
import numpy as np
import operator
from BaseUtils.pretty_util import *
from interval import Interval
import math
from ModelUtil.draw_util import get_correlation
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ModelUtil.draw_util import get_null_pct_by_label
from scipy.stats import ks_2samp

## 特征分析
class FeatureExplore(object):
    def __init__(self, df, max_col_missing_rate=None, max_row_missing_rate=None, max_null_pct_by_sort=None, change_type=True):
        self.log = log
        self.df = df
        if max_col_missing_rate is not None:
            self.max_col_missing_rate = max_col_missing_rate
        else:
            self.max_col_missing_rate = 0.4
        if max_row_missing_rate is not None:
            self.max_row_missing_rate = max_row_missing_rate
        else:
            self.max_row_missing_rate = 0.3
        if max_null_pct_by_sort is not None:
            self.max_null_pct_by_sort = max_null_pct_by_sort
        else:
            self.max_null_pct_by_sort = 0.3
        self.change_type = change_type
        self.real_objects_col = []          # 处理以后不能进行数值型转化的
        self.unreal_onjects_col = []
        self.keep_only_null_cols = True
        self.except_cols_dict = {}
        self.only_single_cols = []
        self.high_corr_df = pd.DataFrame(columns=["col1", "col2", "corr_value"])

    # 定义打印对象时打印的字符串
    def __str__(self, print_all=True):
        if print_all:
            self.log.info(self.__dict__)
        else:
            self.log.info(' '.join(('%s' % item for item in self.__dict__.keys())))

    def go_explore(self):
        """
        对数据整体进行探索，主要目的是以下几点：
        1. 每一列的数据类型，判断是否需要转化
        2. 横向纵向缺失问题
        3. 是否有无效列：一列只有唯一值
        4. 检查时间列
        """
        self.explore_objects()
        self.explore_nulls()
        self.explore_single_value()
        self.explore_datetime_cols()

    def explore_objects(self):
        init_object_cols = list(self.df.columns[self.df.dtypes == 'object'])
        for i, col in enumerate(init_object_cols):
            try:
                if self.change_type:
                    self.df[col] = self.df[col].astype(np.float)
                else:
                    self.unreal_onjects_col.append(col)
            except:
                self.real_objects_col.append(col)
        if not self.change_type:
            print("这些离散变量可以处理:{}".format(self.unreal_onjects_col))
        print("处理以后剩下的离散变量共{}个，分别是：{}\n".format(len(self.real_objects_col), self.real_objects_col))

    def explore_nulls(self, is_show=False):
        # 对不同类别的y值进行缺失值进行分析
        data = self.df.copy()
        null_pct_by_by_label = data.drop('y', axis=1).groupby(data['y']).apply(lambda x: (x.isna().sum()) / (x.shape[0])).T.drop_duplicates(0.0).sort_values(0.0)
        null_pct_by_by_label.columns = ['label_0', 'label_1']
        if is_show:
            get_null_pct_by_label(null_pct_by_by_label)

        null_pct_by_by_label['delta'] = null_pct_by_by_label.apply(lambda x: np.abs(x['label_0'] - x['label_1']),
                                                                   axis=1)
        null_pct_by_by_label = null_pct_by_by_label[null_pct_by_by_label.delta > self.max_null_pct_by_sort]
        self.log.info("对比值超过{}有{}个特征".format(self.max_null_pct_by_sort, len(null_pct_by_by_label.index)))
        self.log.info("分别为{}".format(null_pct_by_by_label.index))
        null_pct_by_by_label.sort_values(by=['delta'], ascending=False)
        print("不同label下的缺失值情况如下\n:{}".format(format_dataframe(null_pct_by_by_label)))


        # 对列的探索
        cols_null_percent = self.df.isnull().sum() / self.df.shape[0]
        to_drop_cols = cols_null_percent[cols_null_percent > self.max_col_missing_rate]
        to_drop_cols = dict(to_drop_cols)
        self.to_drop_cols = sorted(to_drop_cols.items(), key=operator.itemgetter(1), reverse=True)

        # 对行的探索
        rows_null_percent = self.df.isnull().sum(axis=1) / self.df.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > self.max_row_missing_rate]
        to_drop_rows = to_drop_rows.to_dict()
        self.to_drop_rows = sorted(to_drop_rows.items(), key=operator.itemgetter(1), reverse=True)

        self.log.info("缺失值比例>{}的有{}个变量，缺失值比例>{}的总计有{}行".format(
            self.max_col_missing_rate, len(to_drop_cols),
            self.max_row_missing_rate, len(to_drop_rows)))

        result = {
            'dtype': self.df.dtypes,
            'data_cnt': self.df.shape[0],
            'null_cnt': self.df.isnull().sum(),
            'null_ratio': self.df.isnull().sum() / self.df.shape[0]
        }
        result = pd.DataFrame(result, columns=['dtype', 'data_cnt', 'null_cnt', 'null_ratio'])
        result = result[result.null_ratio > self.max_col_missing_rate]
        result = result.sort_values(by=['null_ratio'], ascending=False)
        print("每列缺失值情况如下\n:{}".format(format_dataframe(result)))

    def explore_single_value(self):
        # tip：先计算前100行，或者1000行的集合元素数目
        for i, col in enumerate(self.df.columns):
            try:
                if len(Counter(self.df[col].head(100))) == 1:
                    if len(Counter(self.df[col])) == 1:
                        self.only_single_cols.append(col)
            except:
                pass
        print("总计有{}个变量只有唯一值：{}".format(len(self.only_single_cols), self.only_single_cols))

    def explore_datetime_cols(self):
        datetime_cols = self.df.dtypes[self.df.dtypes == 'datetime64[ns]'].index.tolist()
        if len(datetime_cols) > 0:
            print("共有时间序列{}个，分别是{}\n)".format(len(datetime_cols), datetime_cols))



## 对特征共现姓进行处理
class FeatureCorrExplore(object):
    def __init__(self, df, target_name, is_delete=True, corr_threshold=None, vif_threshold=None):
        self.log = log
        self.df = df
        self.is_delete = is_delete
        self.y = self.df[target_name]
        if corr_threshold is None:
            self.corr_threshold = 0.7
        else:
            self.corr_threshold = corr_threshold
        if vif_threshold is None:
            self.vif_threshold = 10
        else:
            self.vif_threshold = vif_threshold
        self.high_corr_df = pd.DataFrame(columns=["col1", "col2", "corr_value"])
        self.check_df_std()

    def check_df_std(self):
        if self.df.columns[self.df.dtypes == 'object'].size > 0:
            self.log.info("该Dataframe含有字符串类型，无法作相关性计算")
        if self.df.isnull().any().max() == 1:
            self.log.info("该Dataframe含有NULL值，无法计算VIF")

    def get_qutitle00_qutitle99(self):
        q0equalq99_col = set()
        for col in self.df.columns:
            quantile00, quantile99 = self.df[col].quantile([.0, .99])
            if quantile00 == quantile99:
                self.log.info("特征{}, quantitle0={}, quantitle99={}".format(col, quantile00, quantile99))
                q0equalq99_col.add(col)
        return q0equalq99_col

    ## 取得两个变量之间的相关性
    def get_corr(self):
        # 检查变量两两间相关系数
        columns = self.df.columns.tolist()
        col1 = []
        col2 = []
        corr_value = []
        for i in range(len(columns) - 1):
            for j in range(i + 1, len(columns)):
                corr2 = self.df[[columns[i], columns[j]]].corr().iloc[0, 1]
                if abs(corr2) >= self.corr_threshold:
                    col1.append(columns[i])
                    col2.append(columns[j])
                    corr_value.append(corr2)

        # feat_weak_corr内任意两个特征相关性绝对值小于0.7
        self.high_corr_df = pd.DataFrame(data={"col1": col1, "col2": col2, "corr_value": corr_value})
        self.log.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> 相关系数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(format_dataframe(self.high_corr_df))
        get_correlation(self.df, figsize=(20,20))

    ## 删除较小的IV
    def del_high_corr(self, cols_iv_dict):
        deleted_var = []
        high_corr_df = self.high_corr_df.copy()
        high_corr_df['abs_corr_value'] = np.abs(high_corr_df.corr_value)
        high_corr_df = high_corr_df.sort_values(by='abs_corr_value', ascending=False)

        for index, row in high_corr_df.iterrows():
            if row['col1'] == 'y' or row['col2'] == 'y':
                continue
            elif row['col1'] in deleted_var or row['col2'] in deleted_var:
                continue
            elif cols_iv_dict[row['col1']] > cols_iv_dict[row['col2']]:
                deleted_var.append(row['col2'])
            else:
                deleted_var.append(row['col1'])

        if self.is_delete:
            self.df = self.df.drop(deleted_var, axis=1)
        else:
            df1 = self.df.drop(deleted_var, axis=1)
            return df1

    ## 计算各个VIF
    def get_vif(self):
        num_cols = list(range(self.df.shape[1]))
        self.vif_df = pd.DataFrame()
        self.vif_df['vif_value'] = [variance_inflation_factor(self.df.iloc[:, num_cols].values, ix) for ix in range(len(num_cols))]
        self.vif_df['feature'] = self.df.columns
        self.vif_df = self.vif_df.sort_values(by='vif_value', ascending=False)
        print("max(VIF) = {}".format(max(self.vif_df.vif_value)))

    ## 根据VIF删除特征
    def del_vif_col(self):
        if max(self.vif_df.vif_value) < self.vif_threshold:
            self.log.info("没有多重共线性问题!")
        else:
            vifList = self.vif_df.vif_value.tolist()
            maxVifValue = max(vifList)
            X = self.df.copy()
            col = list(range(X.shape[1]))

            while maxVifValue > self.vif_threshold:
                self.log.info('delete var=', X.columns[col[np.argmax(vifList)]], '  ', 'vif=', maxVifValue)
                X = X.drop([X.columns[col[np.argmax(vifList)]]], axis=1)
                col = list(range(X.shape[1]))
                vifList = [variance_inflation_factor(X.iloc[:, col].values, ix) for ix in
                           range(X.iloc[:, col].shape[1])]
                maxVifValue = np.max(vifList)

            if self.is_delete:
                self.df = X.copy()
            else:
                df1 = self.df[X.columns]
                return df1





## 考虑后续变量的PSI统计
class PSIExplore(object):
    def __init__(self, df, serial_cols, dist_cols, bins=5):
        self.df = df
        self.serial_cols = serial_cols
        self.dist_cols = dist_cols
        self.bins_num = int(100/bins)
        self.log = log
        self.IntervalDict = {}


    def get_interval(self, x, null_value=None, has_null=True):

        if has_null:
            x = [i for i in x if not math.isnan(i)]
        elif isinstance(null_value, list):
            x = [i for i in x if x not in null_value]
        elif null_value is not None:
            x = [i for i in x if x != null_value]
        else:
            pass

        pointList = []
        for i in range(4):
            pointList.append(np.percentile(x, self.bins_num * (i + 1)))
        pointList = sorted(list(set(pointList)))
        intervalList = []
        for i in range(len(pointList) + 1):
            if i == 0:
                intervalList.append(Interval(float("-inf"), pointList[0], lower_closed=False, upper_closed=True))
            elif i == len(pointList):
                intervalList.append(Interval(pointList[-1], float("inf"), lower_closed=False, upper_closed=False))
            else:
                intervalList.append(Interval(pointList[i - 1], pointList[i], lower_closed=False, upper_closed=True))
        for i in range(len(x)):
            x[i] = np.argmax([x[i] in k for k in intervalList])
        return x, intervalList


    def main(self):

        changeDF = self.df.copy()
        for c in self.serial_cols:
            self.log.info("Now is dealing the var={}".format(c))
            x = changeDF[c].tolist()
            x, self.IntervalDict[c] = self.get_interval(x)
            changeDF.loc[~changeDF[c].isnull(), c] = x
            changeDF[c] = changeDF[c].fillna(-1)

        transDF = pd.DataFrame(columns=['feature', 'data', 'month'])
        feature = self.serial_cols + self.dist_cols

        for c in feature:
            featureList = [c] * len(self.df)
            dataList = changeDF[c].tolist()
            monthList = changeDF['month'].tolist()
            transDF = pd.concat([transDF, pd.DataFrame(data={'feature': featureList,
                                                             'data': dataList,
                                                             'month': monthList},
                                                       columns=['feature', 'data', 'month'])])

        monthList = sorted(list(set(changeDF['month'].tolist())))
        psiDF = pd.DataFrame(columns=['feature', 'psi', 'month'])
        for i in range(len(monthList)):
            print("Now is dealing the month={}".format(monthList[i]))
            if i <= 2:
                basedf = transDF[transDF.month <= monthList[2]]
                base_cnt = len(self.df[self.df['month'] <= monthList[2]])
            else:
                basedf = transDF[(transDF.month <= monthList[i]) & (transDF.month >= monthList[i - 2])]
                base_cnt = len(self.df[(self.df['month'] <= monthList[i]) & (self.df['month'] >= monthList[i - 2])])
            basedf = basedf.groupby([basedf.feature, basedf.data]).count()
            testdf = transDF[transDF.month == monthList[i]]
            testdf = testdf.groupby([testdf.feature, testdf.data]).count()
            test_cnt = len(self.df[self.df.month == monthList[i]])
            testdf.columns = ['testMonth']
            union_df = pd.merge(basedf, testdf, how='left', left_index=True, right_index=True)
            union_df = union_df.fillna(0)
            union_df['psi'] = union_df.apply(lambda x: np.nan if x.month == 0 or x.testMonth == 0
            else (x.testMonth / test_cnt - x.month / base_cnt) * np.log(x.testMonth / test_cnt - x.month / base_cnt),
                                             axis=1)
            union_df = union_df[['psi']].groupby(level=0).sum().reset_index()
            union_df['month'] = monthList[i]
            psiDF = pd.concat([psiDF, union_df])
            psiDF = pd.pivot_table(psiDF, index=["feature"], values='psi', columns=['month'])
            return psiDF


## 观察连续变量的KS检验
def get_ks2sampe_result(l1, l2):
    s, p = ks_2samp(l1, l2)
    return s,p