#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 14:38
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : ModelEvaluation.py
# @Note    :

import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from prettytable import PrettyTable

from BaseUtils import log

class ModelEvaluation(object):
    def __int__(self, y_true, y_pred, threshold = None, filename = None):
        self.log = log

        if file_name:
            self.filename = file_name
        else:
            self.filename = "draw" + datetime.datetime.today().strftime("%Y%m%d")

        if isinstance(y_true, pd.Series):
            y_true = list(y_true)
        if isinstance(y_pred, pd.Series):
            y_pred = list(y_pred)
        if len(y_true) != len(y_pred):
            print("真实值与预测值长度不一致，请检查!")
            sys.exit(0)
        else:
            self.df = pd.DataFrame(columns=["y_true", "y_pred"], data={"y_true": y_true, "y_pred": y_pred})

        if not threshold:
            self.log.info("暂把阈值设置为0.5")
            self.threshold = 0.5
        else:
            self.threshold = threshold

    @ staticmethod
    def get_roc_curve(y_true, y_pred, file_name=None):
        # 计算真正率和假正率
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        # 计算auc的值
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.figure(figsize=(8, 6))
        # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()

    @ staticmethod
    def get_pr_curve(y_true, y_pred, file_name=None):
        pre1, rec1, thre1 = precision_recall_curve(y_true, y_pred)
        plt.figure()
        plt.title("P-R curve")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(pre1, rec1, color="navy")
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()

    @ staticmethod
    def get_lift_curve(y_true, y_pred, file_name=None):
        result_ = pd.DataFrame(columns=["true_label_col", "pre_prob_col"],
                               data={"true_label_col": list(y_true), "pre_prob_col": list(y_pred)})
        proba_copy = result_.pre_prob_col.copy()
        for i in range(10):
            point1 = np.percentile(result_.pre_prob_col, i * (100 / 10))
            point2 = np.percentile(result_.pre_prob_col, (i + 1) * (100 / 10))
            proba_copy[(result_.pre_prob_col >= point1) & (result_.pre_prob_col <= point2)] = ((i + 1))
        result_['grade'] = proba_copy
        df_gain = result_.groupby(by=['grade'], sort=True).sum() / (len(result_) / 10) * 100
        plt.plot(df_gain['true_label_col'], color='darkorange', lw=2)
        for xy in zip(df_gain['true_label_col'].reset_index().values):
            plt.annotate("%s" % round(xy[0][1], 1), xy=xy[0], xytext=(-20, 10), textcoords='offset points')
        plt.plot(df_gain.index,
                 [sum(result_['true_label_col']) * 100.0 / len(result_['true_label_col'])] * len(df_gain.index),
                 color='blue', lw=2, linestyle='--')
        plt.title('The Change Of Bad Rate')
        plt.xlabel('Quantile')
        plt.ylabel('Bad Rate(%)')
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.legend(loc="lower right")
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()

    @ staticmethod
    def get_kde_curve(y_true, y_pred, file_name=None):
        result_ = pd.DataFrame(columns=["true_label_col", "pre_prob_col"],
                               data={"true_label_col": list(y_true), "pre_prob_col": list(y_pred)})
        plt.figure(figsize=(8, 5))
        sns.kdeplot(result_.loc[result_.true_label_col == 0, 'pre_prob_col'], label='label=0', lw=2, shade=1)
        sns.kdeplot(result_.loc[result_.true_label_col == 1, 'pre_prob_col'], label='label=1', lw=2, shade=1)
        plt.title('Kernel Density Estimation')
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()

    @ staticmethod
    def get_ks_lorenz(y_true, y_pred, filename=None):
        rcParams['figure.figsize'] = 8, 6
        fig, ax = plt.subplots()
        ax.set_xlabel('Percentage', fontsize=15)
        ax.set_ylabel('tpr / fpr', fontsize=15)
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
        ax.set_title("Lorenz curve", fontsize=20)
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        percentage = np.round(np.array(range(1, len(fpr) + 1)) / len(fpr), 4)
        ks_delta = tpr - fpr
        ks_index = ks_delta.argmax()
        # ks 垂直线
        ax.plot([percentage[ks_index], percentage[ks_index]],
                [tpr[ks_index], fpr[ks_index]],
                color='limegreen', lw=2, linestyle='--')
        # ax.text
        ax.text(percentage[ks_index] + 0.02, (tpr[ks_index] + fpr[ks_index]) / 2,
                'ks: {0:.4f}'.format(ks_delta[ks_index]),
                fontsize=13)
        # tpr 线
        ax.plot(percentage, tpr, color='dodgerblue', lw=2, label='tpr')
        # 添加对角线
        ax.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
        # fpr 线
        ax.plot(percentage, fpr, color='tomato', lw=2, label='fpr')
        ax.legend(fontsize='x-large')
        # 显示图形
        if filename is not None:
            plt.savefig(file_path)
        plt.show()

    @ staticmethod
    def get_cm_map(y_true, y_pred, cmap=plt.cm.Blues, filename=None):
        cm = confusion_matrix(y_true, y_pred)
        classes = unique_labels(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if filename is not None:
            plt.savefig(file_path)
        plt.show()