# -*- coding:utf-8 -*-

'''
Created date: 2019-05-21

@author: Cai

note: 
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def bin_class_metrics(y_test, y_pred, y_prob, print_out=True, plot_out=True, model_name="model_name"):
    """计算模型效果指标, 并且绘图 AUC ROC 、 Precision-Recall curves.

    Args:
        model_name="" (str): The model_name="" name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_prob (series): Contains the predicted scores
        print_out (bool): Print the classification metrics and thresholds values
        plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves

    Returns:
        dataframe: The combined metrics in single dataframe
        dataframe: ROC thresholds
        dataframe: Precision-Recall thresholds
        Plot: AUC ROC
        plot: Precision-Recall
        plot: Precision-Recall threshold; also show the number of engines predicted for maintenace per period (queue).
        plot: TPR-FPR threshold

    """

    fpr, tpr, thresh_roc = metrics.roc_curve(y_test, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ks_value = abs(fpr - tpr).max()

    binclass_metrics = {
        'Accuracy': metrics.accuracy_score(y_test, y_pred),
        'Precision': metrics.precision_score(y_test, y_pred),
        'Recall': metrics.recall_score(y_test, y_pred),
        'F1 Score': metrics.f1_score(y_test, y_pred),
        'ROC AUC': metrics.roc_auc_score(y_test, y_prob),
        'KS': ks_value
    }

    df_metrics = pd.DataFrame.from_dict(binclass_metrics, orient='index')
    df_metrics.columns = [model_name]

    if print_out:
        print('-----------------------------------------------------------')
        print(model_name, '\n')
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_test, y_pred))
        print('\nClassification Report:')
        print(metrics.classification_report(y_test, y_pred))
        print('\nMetrics:')
        print(df_metrics)

    if plot_out:

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right", fontsize='small')


    return df_metrics