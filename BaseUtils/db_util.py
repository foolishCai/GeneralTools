# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''


import pandas as pd
from prettytable import PrettyTable
from pyhive import sqlalchemy_presto

def format_for_print(df):
    data = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        data.add_row(row)
    return data


class DbUtil(object):
    def __init__(self):
        self.host = 'prestodb.bigdata.weidai.com'
        self.port = '80'
        self.username = 'Kevin-ma'

    def get_connected(self):
        conn = sqlalchemy_presto.presto.Connection(host=self.host, port=self.port, username=self.username)
        self.con = conn

    def exec_sql(self, sql, pretty=False):
        if len(sql) <= 0:
            return "SQL语句不正确"
        self.get_connected()

        try:
            df = pd.read_sql(sql, self.con)
            if pretty:
                return format_for_print(df)
            else:
                return df
        except:
            return "SQL语句不正确"

    def close_conn(self):
        self.con.close()