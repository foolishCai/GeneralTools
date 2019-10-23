# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''

import seaborn as sns
import matplotlib.pyplot as plt


def get_cor_plot(df, figsize=(8, 6)):
    """ 查看相关系数矩阵"""
    try:
        cor = df.drop('uid')
    except:
        cor = df
    try:
        cor = cor.toPandas()
    except:
        cor = df
    plt.figure(figsize=figsize)
    colormap = plt.cm.viridis
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(cor.astype(float).corr(),
                linewidths=0.1,
                vmax=1.0,
                square=True,
                cmap=colormap,
                linecolor='white',
                annot=True)
    plt.show()