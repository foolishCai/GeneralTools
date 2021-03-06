#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 09:52
@desc:
'''

import sys
sys.path.append("..")

import os
import logging
from configs import log_config

class LogUtil(object):
    def __init__(self, log_name=None):
        if log_name is None:
            log_name = log_config['log_name']
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        # 定义输出格式
        formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')

        # 创建一个文件的handler
        fh = logging.FileHandler(log_config['log_path'] + os.sep + log_name + '.log')
        fh.setFormatter(fmt=formatter)
        self.logger.addHandler(fh)

        # 创建一个控制台输出的handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt=formatter)
        self.logger.addHandler(ch)

    def info(self, msg):
        msg = str(msg)
        self.logger.info(msg)

    def debug(self, msg):
        msg = str(msg)
        self.logger.debug(msg)

    def error(self, msg):
        msg = str(msg)
        self.logger.error("!!!ERROR!!!", msg)


if __name__ == '__main__':

    t = LogUtil()
    t.info('hello 1')
    t.info('hello 2')