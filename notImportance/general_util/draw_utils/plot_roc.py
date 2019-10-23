# -*- coding:utf-8 -*-

'''
Created date: 2019-06-27

@author: Cai

note: 
'''


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



def plot_roc(y_true, y_pred, file_name=None):
    fpr, tpr, threshold = roc_curve(y_true, y_pred) ###计算真正率和假正率
    roc_auc = auc(fpr, tpr) ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(8 ,6))
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