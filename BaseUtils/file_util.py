# -*- coding:utf-8 -*-

'''
Created date: 2019-05-20

@author: Cai

note: 数据,模型加载与保存
'''


import pickle
from BaseUtils.log_util import LogUtil

log = LogUtil()

def dump(df, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(df, file=f)
        log.info("文件保存到：{}".format(file_name))
        try:
            log.info("数据形状：{}".format(df.shape))
            return df
        except:
            pass


def load(file_name):
    with open(file_name, 'rb') as f:
        df = pickle.load(f)
        log.info("加载文件：{}".format(file_name))
        try:
            log.info("数据形状：{}".format(df.shape))
            return df
        except:
            pass
