# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''

from general_util.config import model_config, log

from collections import Counter
import pandas as pd
import numpy as np
import operator


class Explore(object):
    def __init__(self, df, max_col_missing_rate = None, max_row_missing_rate = None):
        self.log = log
        if max_col_missing_rate is not None:
            self.max_col_missing_rate = max_col_missing_rate
        else:
            self.max_col_missing_rate = model_config['max_col_missing_rate']
        if max_row_missing_rate is not None:
            self.max_row_missing_rate = max_row_missing_rate
        else:
            self.max_row_missing_rate = model_config['max_row_missing_rate']
        self.fill_value = model_config['fill_value']
        self.value_counts = model_config['value_counts']

        self.real_objects_col = []  # 处理以后不能进行数值型转化的

        self.keep_only_null_cols = True
        self.except_cols_dict = {}
        self.only_single_cols = []

        self.df = df

    # 定义打印对象时打印的字符串
    def __str__(self, print_all=True):
        if print_all:
            self.log.info(self.__dict__)
        else:
            self.log.info(' '.join(('%s' % item for item in self.__dict__.keys())))


    def go_explore(self, explore=True, fill_na=False):
        '''探索对'''
        if explore:
            self.log.info("数据探索不改变原有数据")
            self.explore_objects()
            to_drop_cols, to_drop_rows = self.explore_nulls()
            null_cols_desc = self.explore_null_desc()
            only_single_value_cols = self.remove_single_value()
            return null_cols_desc, to_drop_cols,  to_drop_rows, only_single_value_cols

        else:
            result = self.df.copy()
            self.log.info("数据探索会改变原有数据")
            self.log.info("原始数据形状为:{}".format(self.df.shape))
            self.explore_objects()
            _,_ = self.explore_nulls()
            self.remove_single_value()
            droped_cols = list(set(list(dict(self.to_drop_cols).keys()) + self.only_single_cols))
            self.log.info("需要删除的列{}个".format(len(droped_cols)))
            result = result.drop(droped_cols, axis=1)
            self.log.info("需要删除的行{}个".format(len(self.to_drop_rows)))
            result = result.drop(list(dict(self.to_drop_rows).keys()), axis=0)
            if fill_na:
                result = self.fill_col_with_value(result, fill_value=self.fill_value, except_cols_dict=self.except_cols_dict)

            self.log.info("处理以后数据的形状为：{}".format(result.shape))
            return result



    def explore_objects(self):

        init_object_cols = list(self.df.columns[self.df.dtypes == 'object'])
        self.log.info("初始共{}个字符型：{}".format(len(init_object_cols), init_object_cols))

        self.log.info("尝试将字符型进行转化......")
        for i, col in enumerate(init_object_cols):
            self.log.info("正在处理第{}个:{}".format(i,col))
            try:
                self.df[col] = self.df[col].astype(np.float)
            except:
                self.log.info("****{}****无法被转化".format(col))
                self.real_objects_col.append(col)
        self.log.info("处理以后剩下的离散变量共{}个，分别是：{}".format(len(self.real_objects_col),self.real_objects_col))
        print(self.df[self.real_objects_col].head())

        for i, col in enumerate(self.real_objects_col):
            self.log.info("-"*15 + str(col) + "-"*15)
            if len(Counter(self.df[col])) <= self.value_counts:
                self.log.info((self.df[col]).value_counts())
            else:
                self.log.info("{}包含了太多不同的值,注意检查:\n {}".format(str(col), self.df[col].value_counts()[:10]))


    def explore_nulls(self):
        '''探索缺失值比例'''

        # 对每列的缺失值进行处理
        cols_null_percent = self.df.isnull().sum()/self.df.shape[0]
        to_drop_cols = cols_null_percent[cols_null_percent >= self.max_col_missing_rate]
        to_drop_cols = dict(to_drop_cols)
        self.to_drop_cols = sorted(to_drop_cols.items(), key=operator.itemgetter(1), reverse=True)

        # 多数的列都是缺失的情况下，对行进行删除
        rows_null_percent = self.df.isnull().sum(axis=1)/self.df.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent >= self.max_row_missing_rate]
        to_drop_rows = dict(to_drop_rows)
        self.to_drop_rows = sorted(to_drop_rows.items(), key=operator.itemgetter(1), reverse=True)

        self.log.info("缺失值比例>={}的有{}个变量，缺失值比例>={}的总计有{}行".format(
                            self.max_col_missing_rate, len(to_drop_cols),
                            self.max_row_missing_rate, len(to_drop_rows)))
        return self.to_drop_cols, self.to_drop_rows

    def explore_null_desc(self):
        '''对每一列进行描述'''
        self.log.info("获取每一列的缺失值及类型描述")
        result = {
            'dtype': self.df.dtypes,
            'null_cnt': self.df.isnull().sum(),
            'data_cnt': self.df.shape[0],
            'null_pct': self.df.isnull().sum() / self.df.shape[0]
        }
        result = pd.DataFrame(result, columns=['dtype', 'null_cnt', 'data_cnt', 'null_pct'])
        if self.keep_only_null_cols:
            result = result[result.null_cnt>0]
            result = result.sort_values(by=['null_pct'], ascending=False)
        return result

    def remove_single_value(self):
        '''若该列只有一个值，移除'''
        self.log.info("检查唯一值的列，数据中有{}个变量".format(self.df.shape[1]))
        for i, col in enumerate(self.df.columns):
            if i%100 == 0:
                self.log.info("正在检查{}-{}".format(self.df.shape[1], i))
            try:
                if len(Counter(self.df[col].head(100)))==1:
                    if len(Counter(self.df[col]))==1:
                        self.only_single_cols.append(col)
            except:
                pass
        self.log.info("总计有{}个变量只有唯一值".format(len(self.only_single_cols)))
        return self.only_single_cols

    def fill_col_with_value(self, df, fill_value=0, except_cols_dict={}):
        self.log.info("缺失值填充处理")
        result = df.copy()

        bad_value = ['', None]

        for col in result.columns:
            if any(result[col].isin(bad_value)):
                result.loc[result[col].isin(bad_value), col] = np.nan

        # 直接用给定值填充，比如0/-99，适合于数值列
        to_fill_cols = list(set(df.columns) - set(except_cols_dict.keys()))
        for col in to_fill_cols:
            null_cnt = result[col].isnull().sum()
            if null_cnt > 0:
                result.loc[result[col].isnull()==True, col] = fill_value
                result[col] = result[col].astype(np.float)

        # 特别的填充
        for col in except_cols_dict.keys():
            result.loc[result[col].isnull() == True, col] = except_cols_dict.get(col)

        return result