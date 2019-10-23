#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: yuwei.chen@yunzhenxin.com
@application:
@time: 2019/10/17 11:49
@desc:
'''

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def get_pr_curve(y_true, y_pred, file_path=None):
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(recall,precision)
    plt.show()
    if file_path is not None:
        plt.savefig(file_path)