#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/11/4 19:21
@desc: 卡方分箱
'''

import pandas as pd
import numpy as np
from configs import log

log = log
# 计算分箱后的好坏样本占比
def GetBinBadRate(df, column, target):
    total = df.groupby([column])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([column])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[column], regroup['bad_rate']))
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return dicts, regroup, overallRate

# 判断箱内bad_rate的单调性
def CheckBadRate(df, column, target):
    df2 = df.copy()
    regroup = GetBinBadRate(df2, column, target)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1] * 1.0 / x[0] for x in combined]
    Monotone = [badRate[i] < badRate[i + 1] and badRate[i] < badRate[i - 1] or badRate[i] > badRate[i + 1] and badRate[i] >badRate[i - 1] for i in range(1, len(badRate) - 1)]
    if True in Monotone:
        return False
    else:
        return True

def split_data(df, col, bins):
    df2 = df.copy()
    N = df2.shape[0]
    n = N//bins
    splitPointIndex = [i * n for i in range(1, bins)]
    rawValues = sorted(list(df2[col]))
    # 取到粗糙卡方划分节点
    splitPoint = [rawValues[i] for i in splitPointIndex]  # 分割点的取值
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint

# 计算卡方值
def chi2(df, total_col, bad_col, overRate):
    df2 = df.copy()
    df2['expected'] = df[total_col].map(lambda x: x*overRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]
    chi2 = sum(chi)
    return chi2

# 得到分箱映射
def AssignBin(x, cutPoints):
    numBin = len(cutPoints) + 1;
    if x <= cutPoints[0]:
        return '(-∞,' + str(cutPoints[0]) + ']'
    elif x > cutPoints[-1]:
        return '(' + str(cutPoints[-1]) + ',+∞]'
    else:
        for i in range(0, numBin-1):
            if x > cutPoints[i] and x <= cutPoints[i+1]:
                return '(' + str(cutPoints[i]) + ',' + str(cutPoints[i+1]) + ']'


# 卡方分箱
def ChiMerge(df, col, target, max_interval=5, minBinRate=0.1):
    df2 = df.copy()
    colsSorted = sorted(list(set(df2[col])))
    N_distinct = len(colsSorted)
    if N_distinct <= max_interval:
        log.info("原始属性{}的取值个数低于max_interval".format(col))
        colsSorted = colsSorted[1:-1]
        colsSorted.extend([float("-inf"), float("inf")])
        return colsSorted
    else:
        if N_distinct > (max_interval/minBinRate):
            log.info("以每箱占比高于{}为标准，该变量分箱数定为={}".format(minBinRate, np.floor(1/minBinRate)))
            split_x = split_data(df2, col, int(np.floor(1/minBinRate)))
            df2['temp'] = df2[col].map(lambda x: AssignBin(x, split_x))
        else:
            df2['temp'] = df2[col]
        binBadRate, regroup, overallRate = GetBinBadRate(df2, 'temp', target)

        # step1：将每个属性列为一组，对各个属性进行排序，然后两两合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]

        # srep2: 循环，不断合并两个组别，直到
        #        2.1 最终分裂出来对分箱数与每箱占比均达标
        #        2.2 每箱同时包含好坏两种样本
        while len(groupIntervals) > max_interval:
            chiList = []
            for k in range(len(groupIntervals)-1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chiList.append(chi2(df2b, 'total', 'bad', overallRate))
            best_comnbined = np.argmin(chiList)
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]
            groupIntervals.remove(groupIntervals[best_comnbined+1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        # 检查每箱是否包含两类样本
        groupedvalues = df2['temp'].map(lambda x: AssignBin(x, cutOffPoints))

        # 检查单调性
        df2['temp_Bin'] = groupedvalues
        (binBadRate, regroup,_) = GetBinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        while (minBadRate == 0 or maxBadRate == 1) and len(cutOffPoints) > 1:
            # 找出全部为好／坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0, 1])].temp_Bin.tolist()
            bin = indexForBad01[0]
            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                # 和前一箱进行合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b, _) = GetBinBadRate(df3, 'temp_Bin', target)
                chisq1 = chi2(df2b, 'total', 'bad', overallRate)
                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b, _) = GetBinBadRate(df3b, 'temp_Bin', target)
                chisq2 = chi2(df2b, 'total', 'bad', overallRate)
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])
             # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            (binBadRate, regroup,_) = GetBinBadRate(df2, 'temp_Bin', target)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

        # 检查分箱后的最小占比
        if minBinRate > 0:
            groupedvalues = df2['temp'].map(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            # value_counts每个数值出现了多少次
            valueCounts = groupedvalues.value_counts().to_frame()
            N = sum(valueCounts['temp'])
            valueCounts['pcnt'] = valueCounts['temp'].map(lambda x: x * 1.0 / N)
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            while minPcnt < minBinRate and len(cutOffPoints) > 2:
                # 占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                else:
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b,_) = GetBinBadRate(df3, 'temp_Bin', target)
                    chisq1 = chi2(df2b, 'total', 'bad', overallRate)
                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b,_) = GetBinBadRate(df3b, 'temp_Bin', target)
                    chisq2 = chi2(df2b, 'total', 'bad', overallRate)

                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
                groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupedvalues
                # value_counts每个数值出现了多少次
                valueCounts = groupedvalues.value_counts().to_frame()
                N = sum(valueCounts['temp'])
                valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N)
                valueCounts = valueCounts.sort_index()
                minPcnt = min(valueCounts['pcnt'])
        return cutOffPoints

