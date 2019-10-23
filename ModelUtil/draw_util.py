#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 14:12
@desc:
'''

import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def get_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Paired, filename=None):

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
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

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight', format='png', dpi=300, pad_inches=0, transparent=True)


def get_correlation(df, figsize=(8, 6)):
    """
    :param df: pandas.Dataframe
    :param figsize:
    :return:
    """

    plt.figure(figsize=figsize)
    colormap = plt.cm.viridis
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.astype(float).corr(),
                linewidths=0.1,
                vmax=1.0,
                square=True,
                cmap=colormap,
                linecolor='white',
                annot=True)
    plt.show()


def get_feature_importance(feature, importance, top=30, filename=None):
    if len(feature) < top:
        top = len(feature)
    d = dict(zip(feature, importance))
    feature_importance_list = sorted(d.items(), key=lambda item: abs(item[1]), reverse=True)
    top_names = [i[0] for i in feature_importance_list][: top]

    plt.figure(figsize=(8, 6))
    plt.title("Feature importances")
    plt.barh(range(top), [d[i] for i in top_names], color="b", align="center")
    plt.ylim(-1, top)
    plt.xlim(min(importance), max(importance))
    plt.yticks(range(top), top_names)
    plt.show()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', format='png', dpi=300, pad_inches=0, transparent=True)
    return feature_importance_list[:top]


def get_ks_lorenz(y, y_pred, title='Lorenz curve', file_path=None):
    rcParams['figure.figsize'] = 8, 8
    fig, ax = plt.subplots()
    ax.set_xlabel('Percentage', fontsize=15)
    ax.set_ylabel('tpr / fpr', fontsize=15)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.set_title(title, fontsize=20)
    fpr, tpr, threshold = roc_curve(y, y_pred)
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
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()


def get_roc_curve(y_true, y_pred, file_name=None):
    fpr, tpr, threshold = roc_curve(y_true, y_pred) ###计算真正率和假正率
    roc_auc = auc(fpr, tpr) ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


def get_pr_curve(y_true, y_pred, file_path=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall,precision)
    plt.show()
    if file_path is not None:
        plt.savefig(file_path)

