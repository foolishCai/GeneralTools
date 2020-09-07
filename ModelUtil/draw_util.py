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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from prettytable import PrettyTable
from scipy.stats import scoreatpercentile
import pandas as pd

def format_dataframe(df):
    data = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        data.add_row(row)
    return data

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



def get_feature_correlation(df, x1_name, x2_name):
    plt.figure(figsize=(8, 5))
    new_data = df[(~df[x1_name].isnull()) & (df[x2_name].isnull())][[x1_name, x2_name]]
    new_data[x1_name] = new_data[x1_name].astype('float')
    new_data[x2_name] = new_data[x2_name].astype('float')
    plt.scatter(new_data[x1_name], new_data[x2_name])
    plt.xlabel(x1_name)
    plt.ylabel(x2_name)
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


def get_ks_lorenz(y_true, y_pred, title='Lorenz curve', file_path=None):
    rcParams['figure.figsize'] = 8, 6
    fig, ax = plt.subplots()
    ax.set_xlabel('Percentage', fontsize=15)
    ax.set_ylabel('tpr / fpr', fontsize=15)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.set_title(title, fontsize=20)
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
    plt.plot(recall, precision)
    plt.show()
    if file_path is not None:
        plt.savefig(file_path)


def get_feature_distribution(df, target_name, feature_name):
    # 单特征分布
    groupby_feature = df.groupby(feature_name).size()
    for a, b in zip(range(len(groupby_feature.index.values.tolist())), groupby_feature.values.tolist()):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=9)
    groupby_feature.plot.bar(title='Distribution of {}'.format(feature_name), width=0.5, figsize=(6, 4))

    # 单特征交互目标变量
    groupby_with_target = df.groupby([feature_name, target_name])[feature_name].count().unstack(target_name).fillna(0)
    groupby_with_target[df[target_name].unique().tolist()].plot.bar(
        title='Distribution of {} by {}'.format(target_name, feature_name), stacked=True, figsize=(6, 4))
    tmp = groupby_with_target.iloc[:, 1] / groupby_with_target.iloc[:, 0]
    for a, b in zip(range(len(tmp.index.values.tolist())), tmp.values.tolist()):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=9)



def get_iv_plt(iv_map_dict):
    plt.figure(figsize=(20, 7))
    plt.hist([i for i in iv_map_dict.values() if i < 5], bins=200)
    plt.show()


def get_report_picture(train_true, train_pred, test_true, test_pred):
    fig = plt.figure(figsize=(12, 12))
    fpr, tpr, threshold = roc_curve(train_true, train_pred)
    roc_auc = auc(fpr, tpr)

    # train-ks
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlabel('Percentage', fontsize=15)
    ax1.set_ylabel('tpr / fpr', fontsize=15)
    ax1.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax1.set_title("Train-ks", fontsize=20)
    percentage = np.round(np.array(range(1, len(fpr) + 1)) / len(fpr), 4)
    ks_delta = tpr - fpr
    ks_index = ks_delta.argmax()
    ax1.plot([percentage[ks_index], percentage[ks_index]],
             [tpr[ks_index], fpr[ks_index]],
             color='limegreen', lw=2, linestyle='--')
    ax1.text(percentage[ks_index] + 0.02, (tpr[ks_index] + fpr[ks_index]) / 2,
             'ks: {0:.4f}'.format(ks_delta[ks_index]),
             fontsize=13)
    ax1.plot(percentage, tpr, color='dodgerblue', lw=2, label='tpr')
    ax1.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
    ax1.plot(percentage, fpr, color='tomato', lw=2, label='fpr')
    ax1.legend(fontsize='x-large')

    # train-auc
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Train ROC curve')
    ax2.legend(loc="lower right")

    fpr, tpr, threshold = roc_curve(test_true, test_pred)
    roc_auc = auc(fpr, tpr)

    # train-ks
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel('Percentage', fontsize=15)
    ax3.set_ylabel('tpr / fpr', fontsize=15)
    ax3.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax3.set_title("Test-ks", fontsize=20)
    percentage = np.round(np.array(range(1, len(fpr) + 1)) / len(fpr), 4)
    ks_delta = tpr - fpr
    ks_index = ks_delta.argmax()
    ax3.plot([percentage[ks_index], percentage[ks_index]],
             [tpr[ks_index], fpr[ks_index]],
             color='limegreen', lw=2, linestyle='--')
    ax3.text(percentage[ks_index] + 0.02, (tpr[ks_index] + fpr[ks_index]) / 2,
             'ks: {0:.4f}'.format(ks_delta[ks_index]),
             fontsize=13)
    ax3.plot(percentage, tpr, color='dodgerblue', lw=2, label='tpr')
    ax3.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
    ax3.plot(percentage, fpr, color='tomato', lw=2, label='fpr')
    ax3.legend(fontsize='x-large')

    # train-auc
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('Test ROC curve')
    ax4.legend(loc="lower right")


def get_badpct_by_time(df, file_name=None):
    plt.figure(figsize=(8, 5))
    df['total'].plot(kind='bar', color="yellow", width=0.4)
    plt.ylabel('总数')
    plt.title('样本随时间分布')
    plt.legend(["总数"])
    df['bad_pct'].plot(color='green', secondary_y=True, style='-o', linewidth=5)
    plt.ylabel('坏样本（比例）')
    plt.legend(["比例"])
    plt.show()
    if file_name:
        plt.savefig(file_name + ".png")


def get_null_pct_by_label(df, file_name=None):
    plt.figure(figsize=(20, 7))
    plt.plot(df['label_0'], 'r', label='y=0')
    plt.plot(df['label_1'], 'g', label='y=1')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()
    if file_name:
        plt.savefig(file_name+".png", dpi=500)

def get_all_null(df, file_name=None):
    plt.figure(figsize=(15, 5))
    plt.scatter(df.num, df.null_pct, c='b')
    plt.xlim([0, len(df.index)])
    plt.xlabel('样本排序')
    plt.ylabel('null值占比')
    plt.title('distribution of null nums')
    plt.show()
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)

def get_correlation(df, file_name=None):
    """
    :param df: pandas.Dataframe
    :param figsize:
    :return:
    """

    plt.figure(figsize=(20, 20))
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
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)



def get_curve(y_true, y_pred, file_name):

        plt.figure(figsize=(12, 6))
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.subplot(121)
        plt.xlabel('Percentage', fontsize=15)
        plt.ylabel('tpr / fpr', fontsize=15)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.title("ks-curve", fontsize=20)

        percentage = np.round(np.array(range(1, len(fpr) + 1)) / len(fpr), 4)
        ks_delta = tpr - fpr
        ks_index = ks_delta.argmax()
        plt.plot([percentage[ks_index], percentage[ks_index]],
                 [tpr[ks_index], fpr[ks_index]],
                 color='limegreen', lw=2, linestyle='--')
        plt.text(percentage[ks_index] + 0.02, (tpr[ks_index] + fpr[ks_index]) / 2,
                 'ks: {0:.4f}'.format(ks_delta[ks_index]),
                 fontsize=13)
        plt.plot(percentage, tpr, color='dodgerblue', lw=2, label='tpr')
        plt.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
        plt.plot(percentage, fpr, color='tomato', lw=2, label='fpr')
        plt.legend(fontsize='x-large')

        plt.subplot(122)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.savefig(file_name)
        plt.show()



def get_cm(y_true, y_pred, thresh, file_name=None):
    cm = confusion_matrix(list(y_true), [int(i>thresh) for i in list(y_pred)])   # 由原标签和预测标签生成混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()


def get_lift_curve(y_true, y_pred, file_name=None):
    result = pd.DataFrame(columns=['target', 'proba'], data={"target": list(y_true), "proba": list(y_pred)})
    result_ = result.copy()
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result_.proba, i * (100 / 10))
        point2 = scoreatpercentile(result_.proba, (i + 1) * (100 / 10))
        proba_copy[(result_.proba >= point1) & (result_.proba <= point2)] = ((i + 1))
    result_['grade'] = proba_copy
    df_gain = result_.groupby(by=['grade'], sort=True).sum() / (len(result) / 10) * 100
    plt.plot(df_gain['target'], color='red')
    for xy in zip(df_gain['target'].reset_index().values):
        plt.annotate("%s" % round(xy[0][1], 2), xy=xy[0], xytext=(-20, 10), textcoords='offset points')
    # plt.plot(df_gain.index, [sum(result['target']) * 100.0 / len(result['target'])] * len(df_gain.index),
    #              color='blue')
    plt.plot(list(df_gain.index), [round(result.target.sum() / len(result), 4) * 100] * len(list(df_gain.index)), color="blue")
    plt.title('Lift Curve')
    plt.xlabel('Decile')
    plt.ylabel('Lift Value')
    plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    if file_name:
        plt.savefig(file_name)
    plt.show()