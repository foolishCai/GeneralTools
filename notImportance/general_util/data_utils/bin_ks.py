# -*- coding:utf-8 -*-

'''
Created date: 2019-05-20

@author: Cai

note: 
'''


# -*- coding: utf-8 -*-
"""
创建KS分箱实验
"""
import pandas as pd


def best_ks_box(data, var_name, target_name, box_num=10):
    data = data[[var_name, target_name]]
    """
    KS值函数
    """

    def ks_bin(data_, limit):
        g = data_.iloc[:, 1].value_counts()[0]
        b = data_.iloc[:, 1].value_counts()[1]
        data_cro = pd.crosstab(data_.iloc[:, 0], data_.iloc[:, 1])
        data_cro[0] = data_cro[0] / g
        data_cro[1] = data_cro[1] / b
        data_cro_cum = data_cro.cumsum()
        ks_list = abs(data_cro_cum[1] - data_cro_cum[0])
        ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
        for i in ks_list_index:
            data_1 = data_[data_.iloc[:, 0] <= i]
            data_2 = data_[data_.iloc[:, 0] > i]
            if len(data_1) >= limit and len(data_2) >= limit:
                break
        return i

    # 测试： ks_bin(data,data.shape[0]/7)

    """
    区间选取函数
    """

    def ks_zone(data_, list_):
        list_zone = list()
        list_.sort()
        n = 0
        for i in list_:
            m = sum(data_.iloc[:, 0] <= i) - n
            n = sum(data_.iloc[:, 0] <= i)
            list_zone.append(m)
        list_zone.append(50000 - sum(list_zone))
        max_index = list_zone.index(max(list_zone))
        if max_index == 0:
            rst = [data_.iloc[:, 0].unique().min(), list_[0]]
        elif max_index == len(list_):
            rst = [list_[-1], data_.iloc[:, 0].unique().max()]
        else:
            rst = [list_[max_index - 1], list_[max_index]]
        return rst

    #    测试： ks_zone(data_,[23])    #左开右闭

    data_ = data.copy()
    limit_ = data.shape[0] / 20  # 总体的5%
    """"
    循环体
    """
    zone = list()
    for i in range(box_num - 1):
        ks_ = ks_bin(data_, limit_)
        zone.append(ks_)
        new_zone = ks_zone(data, zone)
        data_ = data[(data.iloc[:, 0] > new_zone[0]) & (data.iloc[:, 0] <= new_zone[1])]

    """
    构造分箱明细表
    """
    # zone.append(data.iloc[:, 0].unique().max())
    # zone.append(data.iloc[:, 0].unique().min())
    zone.sort()
    # df_ = pd.DataFrame(columns=[0, 1])
    # for i in range(len(zone) - 1):
    #     if i == 0:
    #         data_ = data[(data.iloc[:, 0] >= zone[i]) & (data.iloc[:, 0] <= zone[i + 1])]
    #     else:
    #         data_ = data[(data.iloc[:, 0] > zone[i]) & (data.iloc[:, 0] <= zone[i + 1])]
    #     data_cro = pd.crosstab(data_.iloc[:, 0], data_.iloc[:, 1])
    #     df_.loc['{0}-{1}'.format(data_cro.index.min(), data_cro.index.max())] = data_cro.apply(sum)
    return zone


# data = pd.read_excel('测试1.xlsx')
# var_name = '年龄'
# print(best_ks_box(data, var_name, 5))