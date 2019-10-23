#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 17:10
@desc:
'''


from sklearn.ensemble import RandomForestRegressor

def fill_value_by_RandomForest(df, target_name, is_fill=False):
    # 把数值型特征都放到随机森林里面去
    known_df = df[df[target_name].notnull()]
    unknown_df = df[df[target_name].isnull()]
    y = known_df[target_name]
    x = known_df.drop([target_name], axis=1)
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    # 根据已有数据去拟合随机森林模型
    rfr.fit(x, y)
    # 预测缺失值
    predictedValues = rfr.predict(unknown_df.drop([target_name], axis=1))
    if is_fill:
        df.loc[(df[target_name].isnull()), target_name] = predictedValues
        return rfr, df
    else:
        return rfr, predictedValues