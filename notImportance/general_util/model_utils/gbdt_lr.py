# -*- coding: utf-8 -*-
# create_date: 2019-06-29
# author: buxy
"""

"""
import numpy as np
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# 继承 gbdt，实现transform并能够输出所有叶子节点路径
class GBClassifier(GradientBoostingClassifier, TransformerMixin):

    @staticmethod
    def describe_leave_path(tree, feature_names=None):
        """
        :param tree: 树 sklearn.tree._tree.Tree
        :param feature_names: 真是的变量名
        :return: list，每个叶子节点的路径
        """
        # 确定树的各个属性
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature.copy()
        if feature_names is not None:
            feature = [feature_names[i] for i in feature]
        else:
            feature = ['X[:, {0}]'.format(i) for i in feature]
        threshold = tree.threshold.copy()
        threshold = np.round(threshold, 4)
        # 为每个节点定义好描述，描述为 '父节点的描述—>当前节点id:当前节点描述'
        node_stack = [(0, 'root')]
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        nodes_desc = [str(i) for i in range(n_nodes)]
        while len(node_stack) > 0:
            node_id, node_desc = node_stack.pop()
            nodes_desc[node_id] = node_desc
            if children_left[node_id] == children_right[node_id]:
                # 当两个相等，说明都是-1，该节点为叶子节点
                is_leaves[node_id] = True
                nodes_desc[node_id] = nodes_desc[node_id] + ' -> leave_node:{0}'.format(node_id)
                continue
            if children_left[node_id] != -1:
                left_node_desc_part = '{0} <= {1}'.format(feature[node_id], threshold[node_id])
                node_stack.append(
                    (children_left[node_id], '{0} -> {1}'.format(node_desc, left_node_desc_part)))
            if children_right[node_id] != -1:
                right_node_desc_part = '{0} > {1}'.format(feature[node_id], threshold[node_id])
                node_stack.append(
                    (children_right[node_id], '{0} -> {1}'.format(node_desc, right_node_desc_part)))
        return [nodes_desc[i] for i in range(n_nodes) if is_leaves[i]]

    def describe_trees(self, feature_names=None):
        all_leaves_path = []
        for estimator in self.estimators_.reshape(self.n_estimators_):
            tree_ = estimator.tree_
            all_leaves_path = all_leaves_path + self.describe_leave_path(tree_, feature_names)
        return all_leaves_path

    def transform(self, X):
        n_samples = len(X)
        return self.apply(X).reshape(n_samples, self.n_estimators_)


# 创建数据
X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
X = X.astype(np.float32)
# map labels from {-1, 1} to {0, 1}
labels, y = np.unique(y, return_inverse=True)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# 实现三个实例
gbdt = GBClassifier(learning_rate=0.05, n_estimators=50, subsample=0.8,
                    min_samples_split=5, min_samples_leaf=5, max_depth=2, min_impurity_decrease=0.0001,
                    random_state=1, max_features=0.8, validation_fraction=0.2, n_iter_no_change=10)
onehot = OneHotEncoder(categories='auto')
lr = LogisticRegression()

# 创建pipeline
pipeline = make_pipeline(gbdt, onehot, lr)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict_proba(X_test)[:, 1]

# 准确性
roc_auc_score(y_test, y_pred)
# 假设变量名
feature_names = ['feature_' + str(i) for i in range(10)]
fitted_gbdt = pipeline.steps[0][1]
fitted_gbdt.describe_trees(feature_names=feature_names)
# 叶子节点个数，和逻辑回归的变量数应是一一致的
fitted_lr = pipeline.steps[2][1]
assert len(fitted_lr.coef_[0]) == len(fitted_gbdt.describe_trees(feature_names=feature_names))

# 网格搜索
pipeline.get_params()
param_grid = {
    'gbclassifier__learning_rate': [0.05, 0.10, 0.20],
    'gbclassifier__max_depth': [2, 3, 4],
    'gbclassifier__max_features': [0.6, 0.8],
    'logisticregression__C': [0.5, 1],
    'logisticregression__fit_intercept': [1, 0],
    'logisticregression__penalty': ['l1', 'l2']
}

gsearch = GridSearchCV(pipeline, param_grid=param_grid)

gsearch.fit(X_train, y_train)
y_pred = gsearch.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)