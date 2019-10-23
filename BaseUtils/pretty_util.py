#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 10:40
@desc:
'''

from prettytable import PrettyTable

def format_dataframe(df):
    data = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        data.add_row(row)
    return data