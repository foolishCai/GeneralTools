# -*- coding:utf-8 -*-

'''
Created date: 2019-05-20

@author: Cai

note: 数据,模型加载与保存
'''


import pickle
from BaseUtils.log_util import LogUtil

log = LogUtil()

def dump(data, file_name):
    fw = open(file_name, 'wb')
    pickle.dump(data, fw)
    log.info("文件保存成功！")
    fw.close()



def load(file_name):
    fr = open(file_name, 'rb')
    data = pickle.load(fr)
    fr.close()
    log.info("文件已加载！")
    return data