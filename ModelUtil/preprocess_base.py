# -*- coding:utf-8 -*-
# created_date: 2019-05-15
# author: buxy
"""
数据预处理基类
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        self.X = None
        self.X_dataframe = None
        self.n_samples = None
        self.n_features = None
        self.feature_names = None
        self.y = None
        # transform_data
        self.transform_data = None
        self.transform_dataframe = None
        self.n_transform_samples = None
        self.n_transform_features = None
        self.transform_feature_names = None
        self.transform_y = None
        # transformed_data
        self.transformed_data = None

    @staticmethod
    def to_array(data):
        """
        将输入数据转换成 np.array
        如果转换不成功则直接报错
        :param data: 数据，期望是 array-like, 但有可能不是
        """
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
            except Exception as e:
                raise ValueError('期望数据是 np.ndarray 格式: {0}'.format(e.__str__()))
        return data

    @staticmethod
    def format_data(data):
        """返回数据的n_sample，n_feature，并且当是一维向量时，进行转置"""
        shape = data.shape
        if len(shape) > 1:
            n_samples, n_features = shape
        else:
            n_samples = len(data)
            n_features = 1
            data = data.reshape(n_samples, n_features)
            print('X 可能是单变量，已经自动转置')
        return data, n_samples, n_features

    def fit(self, X, y=None, feature_names=None):
        """
        所有transformer的通用环节(主要用于输入数据格式检查)
        当 X 是Dataframe时，需要存储原始DataFrame，当feature_names为None时，DataFrame的 feature_names 就是真实的feature_names
        :param X: array-like， n_samples * n_features， 如果是单变量，期望是 n * 1 的结构
        :param y: array-like， n_samples，可选 ,这里的y暂时限定只能是符合模型要求的数字，不可以是label
        """
        if isinstance(X, pd.DataFrame):
            self.X_dataframe = X
            self.X_dataframe = self.X_dataframe.reset_index(drop=True)
            if feature_names is None:
                feature_names = X.columns.tolist()
        data = self.to_array(X)
        self.X, self.n_samples, self.n_features = self.format_data(data)
        if y is not None:
            self.y = self.to_array(y)
        if feature_names is not None:
            assert len(feature_names) == self.n_features
            self.feature_names = feature_names
        else:
            self.feature_names = list(range(self.n_features))
        self._fit()
        return self

    def transform(self, X=None, y=None):
        if isinstance(X, pd.DataFrame):
            self.transform_dataframe = X
            self.transform_dataframe = self.transform_dataframe.reset_index(drop=True)
        transform_data = self.to_array(X)
        self.transform_data, self.n_transform_samples, self.n_transform_features = self.format_data(transform_data)
        # 因为支持 dataframe 传输，transform的数据并不要求长度和原始fit数据集长度一致，故不需要assert
        # assert self.n_features == self.n_transform_features, '测试数据与训练数据变量长度不一致'
        if y is not None:
            self.transform_y = self.to_array(y)
        self.transformed_data = self._transform()
        return self.transformed_data

    def _fit(self):
        raise NotImplementedError('该对象还未配置_fit方法')

    def _transform(self):
        raise NotImplementedError('该对象还未配置_transform方法')

    def get_X_dataframe(self, copy=True):
        assert self.X_dataframe is not None, 'The X is not a pd.DataFrame'
        if copy:
            return self.X_dataframe.copy()
        else:
            return self.X_dataframe

    def get_transform_dataframe(self, copy=True):
        assert self.transform_dataframe is not None, 'The transform data is not a pd.DataFrame'
        if copy:
            return self.transform_dataframe.copy()
        else:
            return self.transform_dataframe

    def set_transformed_feature_names(self, transformed_feature_names):
        if not isinstance(transformed_feature_names, list):
            try:
                transformed_feature_names = list(transformed_feature_names)
            except Exception as e:
                raise ValueError("transformed_feature_names can't be a list, {0}".format(e.__str__()))
        self.transformed_feature_names = transformed_feature_names
