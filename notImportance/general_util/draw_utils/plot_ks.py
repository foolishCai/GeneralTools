# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import roc_curve


def plot_ks(preds, labels, n=10, is_score=0, filename=None):
    """
    ks曲线绘图
    preds - 预测列，可以指定是概率，也可以指定是分数
    labels - 实际值
    n = 10 - 切分为几档
    is_score = 0 - 默认是预测概率值
    预测值是得分   is_score=1
    预测值是概率: is_score=0 按照预测概率降序
    """

    pred = preds  # 预测值
    bad = labels  # 默认取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if is_score == 1:
        # 按照得分升序排列
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif is_score == 0:
        # 按照预测概率降序排列
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])

    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good_cnt'] = ksds1.good.cumsum()  # 累计好人
    ksds1['cumsum_bad_cnt'] = ksds1.bad.cumsum()  # 累计坏人-目标
    ksds1['cumsum_good_rate'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)  # 累计好人占比
    ksds1['cumsum_bad_rate'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)  # 累计坏人占比

    ksds = ksds1[['cumsum_good_rate', 'cumsum_bad_rate', 'cumsum_good_cnt', 'cumsum_bad_cnt']]

    ksds['cumsum_good'] = ksds['cumsum_good_rate']
    ksds['cumsum_bad'] = ksds['cumsum_bad_rate']
    ksds['cumsum_good_cnt'] = ksds['cumsum_good_cnt']
    ksds['cumsum_bad_cnt'] = ksds['cumsum_bad_cnt']

    # KS = 累计坏人占比 - 累计好人占比
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    # 按照提供的n=10，切分并且汇总数据
    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good_cnt', 'cumsum_good', 'cumsum_bad_cnt', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good_cnt', 'cumsum_good', 'cumsum_bad_cnt', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # 绘图
    plt.figure(figsize=(8, 6))

    # 累计好人占比
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
             color='blue', linestyle='-', linewidth=2)

    # 累计坏人占比
    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
             color='red', linestyle='-', linewidth=2)
    # ks
    plt.plot(ksds.tile, ksds.ks, label='ks  ',
             color='green', linestyle='-', linewidth=2)

    # KS对应的p值、ks值
    plt.axvline(ks_pop, color='gray', linestyle='--', label='best_pop_value')
    plt.axhline(ks_value, color='green', linestyle='--', label='best_ks')

    # 取最大KS时的累计好人、坏人比例
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--', label='cumsum_good_pct')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red', linestyle='--', label='cumsum_bad_pct')
    plt.title('KS-curve \n(KS=%s ' % np.round(ks_value, 4) +
              'at pop_value=%s)' % np.round(ks_pop, 4), fontsize=15)

    plt.legend(loc='best', frameon=False)
    if filename is None:
        return ksds
    else:
        plt.savefig(filename)




def plot_lorenz(y, y_pred, title='Lorenz curve', file_path=None):
    # plotROC性能上有点尴尬，可以用sklearn里的方法实现
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