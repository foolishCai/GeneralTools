# -*- coding:utf-8 -*-

'''
Created date: 2019-05-28

@author: Cai

note: 单变量分析时的绘图常用法
'''

import matplotlib.pyplot as plt
import seaborn as sns


# 异常值处理，箱线图
def plot_boxplot(df, var):
    if isinstance(df, var):
        var_df = df[var]
        var_df[var].boxplot()
    elif isinstance(var, str):
        var_df = df[[var]]
        var_df[var].boxplot()
    else:
        return "DataFrame 没有这一列"



# 观察两个变量，一个是离散值，一个是连续值
# e.g逻辑回归评分卡模型，预测出来的分数为X，连续值；实际是否逾期未Y，离散值二分量
# 使用sns的核密度函数
def plot_dispersed_continuous(df, target, var):
    facet = sns.FacetGrid(df, hue=target, aspect=4)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(df[var].min(), df[var].max()))
    facet.add_legend()
    plt.show()


## 观察三个变量，其中一个用label标注
# e.g两个模型的分数与实际y的关系
# 散点图，量少的时候使用
def plot_scatter(df, target, score1, score2):
    fig = plt.figure(figsize=(8, 8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('score1')
    plt.ylabel('score2')
    plt.scatter(y=df[df[target] == 0][score1], x=df[df[target] == 0][score2], c='b')
    plt.scatter(y=df[df[target] == 1][score1], x=df[df[target] == 1][score2], c='r')
    plt.show()



# 观察两个变量，两个都是离散型
# e.g.统计每个年龄段相对应的人数与逾期率
def plot_dispersed_dispersed(df, x, y):
    tmp1 = df[[x, y]].groupby([x]).mean().rename(columns={y: 'mean'})
    tmp2 = df[[x, y]].groupby([x]).count().rename(columns={y: 'cnt'})
    tmp = tmp1.join(tmp2)
    x = tmp.index.tolist()
    y1 = tmp.overdue_ratio.tolist()
    y2 = tmp.cnt.tolist()

    plt.rcParams['figure.figsize'] = (6,6)
    fig = plt.figure()
    #画柱形图
    ax1 = fig.add_subplot(111)
    ax1.bar(x, y1,alpha=.4,color='g',width=0.45)
    ax1.set_ylabel('overdue_ratio',fontsize='15')
    #画折线图
    ax2 = ax1.twinx()
    for a,b in zip(x,y2):
        ax2.text(a, b+0.05, '%.0f' %b, ha='center', va= 'bottom',fontsize=11)
    ax2.plot(x, y2, 'r',ms=10)
    ax2.set_ylabel('cnt',fontsize='15')
    if len(x)>8:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(90)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(90)
    plt.show()





