#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 14:35
@desc: 无监督分箱，傻瓜称呼：等频
'''

import pandas as pd
from scipy import stats
import numpy as np

def get_qcut_bin(Y, X, n = 10):
    r = 0
    total_bad = Y.sum()
    total_good = Y.count()-total_bad
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n,duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
    d3['min_' + X.name] = d2.min().X
    d3['max_' + X.name] = d2.max().X

    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3[Y.name + '_rate'] = d2.mean().Y
    d3['badattr']=d3[Y.name]/total_bad
    d3['goodattr']=(d3['total']-d3[Y.name])/total_good
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min_' + X.name)).reset_index(drop = True)
    cut = []
    for i in range(1,n+1):
        qua =X.quantile(i/(n+1))
        cut.append(round(qua,4))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe


# 等频分箱优化版：根据分为点进行分箱
# 解决问题：不会出现同一个数值出现在两个箱内
def get_better_quct_bin(Y, X, n=10):

    pass