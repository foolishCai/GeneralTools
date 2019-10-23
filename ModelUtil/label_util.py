# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''

from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(object):
    '''对多个变量进行Encode'''

    def __init__(self, df, columns=None):
        if columns is None:
            print("-"*15, "没有指定转化列，将对所有dtype=object进行转变")
        self.df = df
        self.columns = list(self.df.columns[self.df.dtypes == 'object'])

    def transform(self):
        output = self.df[self.columns].copy()
        for col in self.columns:
            output[col] = LabelEncoder().fit_transform(output[col].map(str))
        return output
