# -*- coding:utf-8 -*-

'''
Created date: 2019-05-21

@author: Cai

note: xgbt模板
'''

"""
第一部分应该根据决策树，得到数据的叶子节点编码，返回数据
第二部分把train,test数据分别转化为woe形式
第三部分，针对节点，拟合LR(或者其他）模型
第三部分,并且做出评估
"""

import numpy as np
import graphviz
import sys

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from general_util.base_utils.base_log import BaseLog


class DecisionTreeClassifierNode(DecisionTreeClassifier):

    def __init__(self, tree_features=None, criterion='entropy', splitter='best', max_depth=3, min_samples_split=2000,
                 min_samples_leaf=800, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=59, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, class_weight=None, presort=False):
        super(DecisionTreeClassifierNode, self).__init__(
            criterion=criterion, splitter=splitter, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, random_state=random_state,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, class_weight=class_weight, presort=presort
        )
        self.__tree_features = tree_features
        self.log = BaseLog()

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):

        if self.__tree_features != None:
            """只以特定的列拟合决策树模型"""
            if not isinstance(self.__tree_features, list):
                try:
                    self.__tree_features = list(self.__tree_features)
                except:
                    self.log.info("tree_feature参数必须是list")
                    sys.exit(0)
            else:
                X = X[self.__tree_features]

        self.tree_clf = super(DecisionTreeClassifier, self).fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)

        self.log.info("树模型如下：\n {}".format(self.tree_clf))

        # 决策树绘图
        dot_data = tree.export_graphviz(self.tree_clf, out_file=None,
                                        feature_names=X.columns,
                                        class_names=['0', '1'],
                                        filled=True, rounded=True,
                                        special_characters=True)

        self.graph = graphviz.Source(dot_data)
        self.log.info("可以通过访问self.graph绘制决策树")

        # 树模型的参数
        self.n_features_ = self.tree_clf.n_features_
        self.classes_ = self.tree_clf.classes_
        self.class_id = dict(zip(self.classes_, range(len(self.classes_))))
        self.tree_ = self.tree_clf.tree_  # 树模型
        self.children_left = self.tree_.children_left  # 左节点
        self.children_right = self.tree_.children_right  # 右节点
        self.feature = self.tree_.feature
        self.threshold = self.tree_.threshold
        self.leaves_id = [i for i in range(self.tree_.node_count) if \
                          self.children_left[i] == -1 and self.children_right[i] == -1]  # 叶子节点
        self.leaves_class = [self.classes_[i] for i in self.tree_.value.argmax(axis=2).ravel()]
        self.leaves_class_id = [self.class_id[i] for i in self.leaves_class]

        self.log.info('决策树有{}个叶子节点'.format(len(self.leaves_id)))

        return self

    def transform(self, X):
        """
        获取每个观测所属的节点

        :param X: DataFrame
        :return:
        """
        if self.__tree_features != None:
            sub_X = X[self.__tree_features]
        else:
            sub_X = X

        leaf_ids = np.apply_along_axis(self._get_data_leaf_id, 1, sub_X)
        X['_leaf_ids'] = leaf_ids
        return X

    def _get_data_leaf_id(self, x):
        """
        获取每个观测所属的树的节点
        :param x:
        :return:
        """
        curr_node_id = 0
        while self.children_left[curr_node_id] != -1 and self.children_right[curr_node_id] != -1:
            dim = self.feature[curr_node_id]
            val = self.threshold[curr_node_id]
            if x[dim] <= val:
                curr_node_id = self.children_left[curr_node_id]
            else:
                curr_node_id = self.children_right[curr_node_id]
        return curr_node_id
